---
title: Incremental Data Processing with Structured Streaming and Auto Loader
categories: [Data Engineering, Databricks]
tags: [databricks, structured streaming, auto loader, checkpoints, bronze silver gold, delta lake]
author: 'Babin'
pin: false
math: false
image:
  path: /assets/img/databricks-structured-streaming.png
  alt: Structured Streaming concepts from the course notes
---

A data stream is any source that grows over time: new files arriving in cloud storage, events published to Kafka, change-data-capture records, or rows appended to a Delta table.

Spark Structured Streaming lets engineers describe transformations using familiar DataFrame or SQL operations while Spark tracks incremental progress.

![Structured Streaming concepts](/assets/img/databricks-structured-streaming.png)
_Structured Streaming treats an evolving source as an unbounded table_

## The Unbounded Table Model

Structured Streaming represents an incoming stream as an unbounded table. New events behave like new rows. A query processes those rows in small batches and writes results to a sink.

Common sources include:

- Cloud files
- Delta tables
- Kafka and compatible messaging systems
- Rate and socket sources for testing

Common sinks include Delta tables, files, Kafka, memory, and custom `foreachBatch` logic.

## Reading and Writing a Stream

A Delta source can be read with:

```python
orders_stream = spark.readStream.table("orders_bronze")
```

The result is a streaming DataFrame. Transformations are declared normally:

```python
valid_orders = orders_stream.filter("quantity > 0")
```

Write the result with its own checkpoint:

```python
(
    valid_orders.writeStream
    .format("delta")
    .option(
        "checkpointLocation",
        "/Volumes/training/checkpoints/orders_silver"
    )
    .outputMode("append")
    .toTable("orders_silver")
)
```

Each streaming write requires a unique checkpoint location. Sharing checkpoints between independent queries corrupts their progress tracking.

## Why Checkpoints Matter

A checkpoint stores the stream's progress and state. If compute fails, the query can restart and continue from the recorded offsets rather than processing everything from the beginning.

The checkpoint belongs to the query. Deleting it changes the query's memory of what has already been processed and may cause reprocessing.

> Treat checkpoints as operational state. Do not casually delete, move, or reuse them.
{: .prompt-warning }

## Trigger Modes

A trigger controls when data is processed.

| Trigger | Behavior |
|---|---|
| Default processing time | Runs micro-batches as quickly as practical |
| Fixed interval | Runs a micro-batch at a configured interval |
| `availableNow` | Processes all currently available data in one or more batches, then stops |

`availableNow` is useful for scheduled incremental jobs. It combines streaming progress tracking with finite job execution.

## Output Modes

The output mode controls what each trigger writes:

- **Append** writes only finalized new rows.
- **Complete** rewrites the full result table, commonly for supported aggregations.
- **Update** writes rows changed since the previous trigger where supported.

The transformation determines which modes are valid. Stateful aggregation also requires careful treatment of event time and late data.

## Auto Loader

Auto Loader incrementally discovers files in cloud object storage using the `cloudFiles` source:

```python
orders_raw = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "parquet")
    .schema(order_schema)
    .load("/Volumes/training/raw/orders")
)
```

Auto Loader is preferable to repeatedly listing a large directory and manually tracking processed filenames. It supports schema inference and evolution options, but production pipelines should define how unexpected columns and malformed records are handled.

## Building Bronze, Silver, and Gold

### Bronze

Bronze preserves source fidelity and ingestion metadata:

```python
from pyspark.sql import functions as F

orders_bronze = (
    orders_raw
    .withColumn("ingested_at", F.current_timestamp())
    .withColumn("source_file", F.input_file_name())
)
```

This layer supports replay and troubleshooting.

### Silver

Silver applies validation, standardization, deduplication, and enrichment. A streaming fact source can be joined with a static lookup:

```sql
SELECT
  o.order_id,
  o.customer_id,
  c.first_name,
  c.last_name,
  o.quantity
FROM orders_bronze_stream o
JOIN customers_lookup c
  ON o.customer_id = c.customer_id
WHERE o.quantity > 0;
```

### Gold

Gold contains business-level aggregates:

```sql
SELECT
  customer_id,
  date_trunc('DAY', order_timestamp) AS order_date,
  sum(quantity) AS books_ordered
FROM orders_silver_stream
GROUP BY customer_id, date_trunc('DAY', order_timestamp);
```

For a scheduled aggregate, `availableNow` can process new silver data and update the gold result.

## Exactly-Once Reasoning

Structured Streaming combines source offsets, checkpoints, write-ahead information, and compatible sinks to provide strong processing guarantees. In practice, the whole pipeline must support replay:

- The source must be repeatable.
- Transformations should be deterministic where possible.
- The sink must handle retries safely.
- External side effects need idempotency keys or deduplication.

`foreachBatch` is powerful, but exactly-once behavior is not automatic when the batch writes to an external API or database.

## Operational Checklist

- Give every query a unique checkpoint.
- Store checkpoints in durable governed storage.
- Monitor input rate, processing rate, batch duration, and failures.
- Define a malformed-record strategy.
- Include source file and ingestion timestamps.
- Test restarts and replay behavior.
- Use watermarks for stateful event-time processing where appropriate.
- Avoid unbounded state growth.

## Source Notes

This article was developed from my Notion notes: [4. Incremental Data Processing](https://app.notion.com/p/05774d28716e4abbb27f27e32089369e).

