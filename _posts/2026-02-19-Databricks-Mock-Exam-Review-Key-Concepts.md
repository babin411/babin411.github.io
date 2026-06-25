---
title: Databricks Mock Exam Review - Key Concepts and Common Traps
categories: [Data Engineering, Databricks]
tags: [databricks, mock exam, delta lake, workflows, auto loader, sql warehouse, certification]
author: 'Babin'
pin: false
math: false
image:
  path: /assets/img/databricks-mock-exam-review.png
  alt: Databricks mock exam question from the Notion notes
---

This review distills the recurring ideas in my Databricks mock-exam notes. It does not reproduce the complete question bank. Instead, it groups the explanations into concepts that can be applied to new scenarios.

![Example mock exam question](/assets/img/databricks-mock-exam-review.png)
_A question screenshot preserved locally from the Notion mock-exam notes_

## Compute Startup and Cluster Pools

Cluster pools keep cloud instances ready for classic compute, reducing the time needed to acquire new virtual machines. They are useful when jobs require faster startup but serverless compute is not being used.

Remember the distinction:

- A pool reduces instance acquisition time.
- Autoscaling changes the number of workers during execution.
- Auto termination stops idle interactive compute.
- A job cluster provides isolated compute for a job run.

These features solve different problems.

## Scale Up vs Scale Out

When one query is slow because it needs more resources, scaling up to a larger size may help. When many queries run concurrently, scaling out by adding clusters or increasing concurrency capacity is usually the relevant direction.

Always identify whether the problem is single-query performance or concurrent demand.

## Delta Lake and ACID

Delta Lake improves reliability by adding ACID transactions and a transaction log to data stored in the lake.

This supports:

- Consistent reads
- Concurrent writes
- Table history
- Updates and deletes
- `MERGE`
- Batch and streaming access

A Delta table is a directory containing data files and transaction-log information, not one monolithic file.

## `VACUUM` and History

`VACUUM` removes old data files that are no longer referenced by the current table state and are older than the retention threshold.

The key consequence is loss of access to older versions that require those files. Never reduce retention casually in a production table.

Do not confuse:

- `DESCRIBE HISTORY`, which displays operations
- Time travel, which reads an older version
- `RESTORE`, which creates a new version based on an old state
- `VACUUM`, which deletes unreferenced old files

## Git Integration

The notes use the older Databricks Repos name. Current Databricks interfaces may call these Git folders.

Git integration supports branches, pull and push operations, and collaboration with a remote repository. Some advanced Git operations may still be performed in the remote provider or local Git tooling rather than entirely through the Databricks UI.

The exam concept is the advantage of real source control over notebook-only revision history.

## Auto Loader

Auto Loader is designed for incremental file ingestion from cloud storage. It tracks discovered files and processes newly arriving data efficiently.

It is a strong answer when a scenario asks to:

- Ingest cloud files incrementally
- Avoid repeatedly processing old files
- Handle continuously arriving file data
- Combine file ingestion with Structured Streaming

It is not a replacement for Kafka when the source is an event stream rather than files.

## Streaming Output Modes

Append mode writes newly finalized rows. Complete mode writes the entire result table for each trigger and is commonly associated with supported aggregations.

Do not select an output mode from the target layer name alone. Determine whether the query is append-only, stateful, or recalculating an aggregate.

## Data Quality in Declarative Pipelines

Pipeline expectations can retain, drop, or fail records depending on the configured rule. The pipeline graph and data-quality metrics help identify where records were rejected.

When a scenario asks where rows are being dropped, inspect expectation metrics on the relevant pipeline datasets rather than manually comparing every source file.

## Workflow Dependencies

A dependency graph describes execution order. If task B depends on task A, task A runs first.

For confusing questions, draw arrows:

```text
new_validation_task -> existing_publish_task
```

The task on the left must complete before the task on the right can start.

## SQL Warehouse Cost Controls

Useful cost controls include:

- Auto stop
- Appropriate warehouse size
- Scaling limits
- Time-bound refresh schedules
- Avoiding unnecessarily frequent dashboard refreshes

An alert or dashboard requirement should not automatically imply always-on compute. Match availability and refresh frequency to the actual service level.

## Alerts

Databricks SQL alerts evaluate saved-query results against a threshold. Notification destinations can include email and configured webhooks.

A typical quality alert starts with a query:

```sql
SELECT count(*) AS null_count
FROM customers
WHERE customer_id IS NULL;
```

The alert triggers when `null_count` reaches the defined threshold.

## Governance and Ownership

Ownership determines who can manage an object. Administrative roles may be required to transfer ownership when the current owner is unavailable.

For grants, use the narrowest required privilege and remember parent-level requirements:

```sql
GRANT USE CATALOG ON CATALOG customers TO `data_engineers`;
GRANT USE SCHEMA ON SCHEMA customers.curated TO `data_engineers`;
GRANT SELECT ON TABLE customers.curated.orders TO `data_engineers`;
```

Broad `ALL PRIVILEGES` access is appropriate only when the scenario truly requires full management.

## Final Review Method

For every mock question:

1. Name the tested feature.
2. State the requirement in one sentence.
3. Eliminate options that solve a different problem.
4. Check scope, lifecycle, cost, and governance.
5. Verify older terminology against current documentation.

That method is more durable than memorizing an answer letter.

## Source Notes

This article was developed from my Notion notes: [Mock Exam - 1](https://app.notion.com/p/5751c573ab1e45f5919abfa721d9a820).

