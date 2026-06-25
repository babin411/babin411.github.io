---
title: Databricks Lakehouse and Delta Lake Fundamentals
categories: [Data Engineering, Databricks]
tags: [databricks, delta lake, acid, time travel, parquet, merge, optimize, views]
author: 'Babin'
pin: false
math: false
image:
  path: /assets/img/databricks-delta-lake.png
  alt: Delta Lake architecture from the course notes
---

Delta Lake is the storage layer that gives lakehouse tables reliable transactions, schema controls, version history, and efficient data management while retaining data in cloud object storage.

It is not a database service or a separate storage medium. A Delta table is built from data files, normally Parquet, plus a transaction log that records which files belong to each table version.

![Delta Lake architecture](/assets/img/databricks-delta-lake.png)
_Delta Lake adds a transaction log and table semantics to files in object storage_

## The Delta Transaction Log

Every Delta table contains a `_delta_log` directory. The log records committed changes to the table, including files added or removed, metadata updates, and operation details.

Readers use the log to construct a consistent snapshot. A partially written data file is not visible as part of the table until its transaction commits.

This design supports the ACID properties:

- **Atomicity**: a transaction succeeds completely or is not committed.
- **Consistency**: committed changes preserve the table's rules.
- **Isolation**: concurrent operations see valid snapshots.
- **Durability**: committed changes remain in durable storage.

## Creating and Inspecting a Delta Table

Databricks uses Delta as the default table format in many configurations:

```sql
CREATE TABLE employees (
  id INT,
  name STRING,
  salary DOUBLE
);

INSERT INTO employees VALUES
  (1, 'Adam', 3500.0),
  (2, 'Sarah', 4020.5),
  (3, 'John', 2999.3);
```

Useful inspection commands include:

```sql
DESCRIBE DETAIL employees;
DESCRIBE HISTORY employees;
SHOW CREATE TABLE employees;
```

`DESCRIBE DETAIL` exposes properties such as location, format, file count, and partition columns. `DESCRIBE HISTORY` shows table operations and versions.

## Managed and External Tables

A managed table delegates both metadata and data lifecycle management to the catalog. An external table registers metadata over data stored in a separately managed location.

The practical difference becomes clear when a table is dropped:

- Dropping a managed table normally removes its managed data.
- Dropping an external table removes the catalog entry but leaves the underlying files.

Unity Catalog should be used to govern both patterns through managed storage, external locations, and appropriate permissions.

## Updating Data with Delta

Delta supports familiar DML operations:

```sql
UPDATE employees
SET salary = salary * 1.05
WHERE id = 2;

DELETE FROM employees
WHERE id = 3;
```

For upserts, use `MERGE`:

```sql
MERGE INTO employees AS target
USING employee_updates AS source
ON target.id = source.id
WHEN MATCHED THEN
  UPDATE SET *
WHEN NOT MATCHED THEN
  INSERT *;
```

`MERGE` is central to change-data-capture pipelines, deduplication, and slowly changing dimensions.

## Time Travel and Restore

Because the transaction log tracks versions, previous snapshots can be queried:

```sql
SELECT *
FROM employees VERSION AS OF 2;
```

Or by timestamp:

```sql
SELECT *
FROM employees
TIMESTAMP AS OF '2025-10-20T10:00:00Z';
```

To restore a table:

```sql
RESTORE TABLE employees TO VERSION AS OF 2;
```

Time travel depends on retaining the required log and data files. `VACUUM` can permanently delete old files, so retention settings must be treated as a recovery policy rather than a housekeeping detail.

## Schema Enforcement and Evolution

Delta validates incoming data against the target schema. This prevents accidental writes with incompatible types or unexpected columns.

When a legitimate schema change is required, use an intentional evolution mechanism such as:

```sql
ALTER TABLE employees ADD COLUMNS (department STRING);
```

Automatic schema evolution can be useful for controlled ingestion, but enabling it broadly may hide upstream data-contract problems.

## Optimizing Delta Tables

Small files increase metadata and scheduling overhead. Databricks provides optimization features to compact files and improve data skipping.

```sql
OPTIMIZE employees;
```

For frequently filtered columns, clustering or data-layout techniques can reduce the amount of data read. The correct strategy depends on table size, update frequency, and query patterns.

Avoid partitioning by high-cardinality columns such as unique identifiers. Date, region, or tenant keys can be useful when they align with common filters and produce reasonably sized partitions.

## Views in Databricks

Views store query logic rather than independent data.

| View type | Lifetime | Scope |
|---|---|---|
| Stored view | Until dropped | Available through its catalog and schema |
| Temporary view | Current Spark session | Current notebook or session |
| Global temporary view | Compute lifetime | Sessions attached to the same compute |

Examples:

```sql
CREATE VIEW high_earning_employees AS
SELECT * FROM employees WHERE salary >= 4000;

CREATE TEMP VIEW employee_departments AS
SELECT DISTINCT department FROM employees;
```

Stored views are useful for reusable semantic logic and access control. Temporary views are useful for intermediate transformations.

## What to Remember

- Delta tables combine Parquet data files with a transaction log.
- Readers use committed snapshots rather than scanning arbitrary files.
- `MERGE`, time travel, schema enforcement, and table history are core capabilities.
- `VACUUM` affects recoverability.
- Managed and external tables have different lifecycle behavior.
- Optimization should follow actual file and query patterns.
- Views store reusable query definitions, not copies of the source data.

## Source Notes

This article was developed from my Notion notes: [2. Databricks Lakehouse Platform](https://app.notion.com/p/ca755fa673534c11a96fd36a098ca673).

