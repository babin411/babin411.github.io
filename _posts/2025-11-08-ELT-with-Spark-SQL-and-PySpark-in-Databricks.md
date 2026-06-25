---
title: ELT with Spark SQL and PySpark in Databricks
categories: [Data Engineering, Databricks]
tags: [databricks, spark sql, pyspark, elt, ctas, json, csv, udf, higher order functions]
author: 'Babin'
pin: false
math: false
image:
  path: /assets/img/databricks-elt-spark-sql.png
  alt: Querying files with Spark SQL from the course notes
---

Databricks supports an ELT workflow in which raw data is loaded into the lakehouse and transformed using Spark SQL or PySpark. Engineers can query files directly, register external data, create Delta tables, work with nested structures, and move between SQL and DataFrame APIs.

![Querying files with Spark SQL](/assets/img/databricks-elt-spark-sql.png)
_Spark SQL can query supported file formats directly before the data is loaded into managed tables_

## Querying Files Directly

Spark SQL can query a path using the format name:

```sql
SELECT *
FROM json.`/Volumes/training/raw/customers`;
```

The path is wrapped in backticks. A directory can contain multiple files when they share a compatible format and schema.

This works especially well with self-describing formats such as JSON and Parquet. CSV normally requires options for headers, delimiters, quoting, and schema.

For raw inspection, text-based data can be read as strings:

```sql
SELECT *
FROM text.`/Volumes/training/raw/events`;
```

Binary files can be loaded with the `binaryFile` source when metadata and binary content are required for images or other unstructured data.

## CTAS: Create Table as Select

`CREATE TABLE AS SELECT`, or CTAS, creates a table from a query:

```sql
CREATE TABLE customers_bronze AS
SELECT *
FROM json.`/Volumes/training/raw/customers`;
```

CTAS infers the target schema from the query result. It is convenient when the source can already be read correctly.

For CSV, first create a temporary view with the required reader options, then create the Delta table:

```sql
CREATE TEMP VIEW customers_csv
USING csv
OPTIONS (
  path '/Volumes/training/raw/customers-csv',
  header 'true',
  inferSchema 'true',
  delimiter ','
);

CREATE TABLE customers_bronze AS
SELECT * FROM customers_csv;
```

For production ingestion, an explicitly defined schema is generally safer than `inferSchema`.

## External Tables

An external table can point to existing files without copying them:

```sql
CREATE TABLE raw_customers
USING CSV
OPTIONS (
  path '/Volumes/training/raw/customers-csv',
  header 'true'
);
```

This table reflects the source format. It does not automatically become Delta simply because it is registered in Databricks.

External tables are useful for discovery and interoperability, but loading curated data into Delta enables transactions, history, and stronger table management.

## Working with JSON

Semi-structured data often contains structs and arrays. Spark SQL supports field access and parsing functions:

```sql
SELECT
  customer_id,
  profile:first_name::STRING AS first_name,
  profile:last_name::STRING AS last_name
FROM customers_bronze;
```

To turn an array into rows:

```sql
SELECT
  order_id,
  item.book_id,
  item.quantity
FROM orders
LATERAL VIEW explode(books) AS item;
```

This is useful when an order contains an array of purchased books and downstream reporting needs one row per item.

## Higher-Order Functions

Higher-order functions transform arrays without always exploding them.

Filter array elements:

```sql
SELECT
  order_id,
  filter(books, book -> book.quantity >= 2) AS multiple_copies
FROM orders;
```

Transform each element:

```sql
SELECT
  order_id,
  transform(books, book -> book.subtotal * 0.9) AS discounted_subtotals
FROM orders;
```

Other useful functions include `exists`, `forall`, `aggregate`, and `zip_with`. These functions preserve the nested structure while applying logic to its contents.

## Pivoting Data

Pivoting turns row values into columns:

```sql
SELECT *
FROM (
  SELECT customer_id, order_date, quantity
  FROM orders
)
PIVOT (
  sum(quantity)
  FOR order_date IN ('2025-11-01', '2025-11-02')
);
```

Pivoted output can be convenient for reports, but it is usually less flexible than normalized data for reusable pipelines.

## SQL User-Defined Functions

A SQL UDF packages reusable logic:

```sql
CREATE OR REPLACE FUNCTION email_domain(email STRING)
RETURNS STRING
RETURN split(email, '@')[1];
```

It can then be used like a built-in function:

```sql
SELECT
  customer_id,
  email_domain(email) AS domain
FROM customers;
```

Prefer built-in Spark SQL expressions where possible because they are well understood by the optimizer. A SQL UDF is appropriate when it makes repeated domain logic clearer and more consistent.

## Moving Between SQL and PySpark

Tables and temporary views form a bridge between APIs:

```python
orders_df = spark.table("orders_bronze")

clean_df = (
    orders_df
    .filter("quantity > 0")
    .select("order_id", "customer_id", "quantity")
)

clean_df.createOrReplaceTempView("orders_clean")
```

The temporary view can be queried from SQL:

```sql
SELECT customer_id, sum(quantity) AS books_ordered
FROM orders_clean
GROUP BY customer_id;
```

## Practical ELT Pattern

A reliable Databricks ELT flow often follows these steps:

1. Land immutable source files in governed cloud storage.
2. Read them with an explicit schema and source-specific options.
3. Preserve ingestion metadata such as source file and arrival time.
4. Write a bronze Delta table.
5. Clean, validate, and standardize data into silver tables.
6. Build business-level gold tables or views.
7. Test row counts, uniqueness, null handling, and reconciliation rules.

## Source Notes

This article was developed from my Notion notes: [3. ELT with SparkSQL and Python](https://app.notion.com/p/9cd85a41c97d42bbb90b605b790e489e).

