---
title: Production Pipelines, Workflows, and Databricks SQL
categories: [Data Engineering, Databricks]
tags: [databricks, lakeflow, delta live tables, workflows, jobs, sql warehouse, dashboards, alerts]
author: 'Babin'
pin: false
math: false
image:
  path: /assets/img/databricks-production-pipelines.png
  alt: Databricks production pipeline graph from the course notes
---

Moving a notebook into production requires more than scheduling it. A production pipeline needs explicit dependencies, data-quality rules, repeatable configuration, monitoring, notifications, and a clear compute strategy.

The original notes use the name **Delta Live Tables (DLT)**. Databricks now presents this capability as **Lakeflow Declarative Pipelines**. Older DLT terminology still appears in learning material and code examples, so it is useful to recognize both names.

![Databricks declarative pipeline graph](/assets/img/databricks-production-pipelines.png)
_A declarative pipeline graph makes bronze, silver, and gold dependencies visible_

## Declarative Pipelines

A declarative pipeline defines datasets and their relationships. The platform determines execution order from the dependency graph.

A typical design includes:

- Bronze streaming tables for raw ingestion
- Silver tables for validation and enrichment
- Gold tables or materialized views for business aggregates
- Expectations that measure or enforce data quality
- An event log for operational analysis

The main benefit is not shorter SQL. It is the managed dependency graph, observability, and data-quality framework.

## Data Quality Expectations

Expectations declare valid-data rules. Depending on the policy, a violation can:

- Be recorded while the row is retained
- Cause the invalid row to be dropped
- Fail the pipeline update

Conceptually:

```sql
CONSTRAINT valid_order_id EXPECT (order_id IS NOT NULL)
```

Choose behavior according to the contract:

- Retain and measure when profiling a new source.
- Drop when invalid records are understood and recoverable.
- Fail when publishing incorrect data would be worse than delaying the pipeline.

Quality rules should also be observable through metrics and alerts.

## Triggered and Continuous Pipelines

A **triggered** pipeline processes available data and stops. It is appropriate for hourly, daily, or event-driven schedules.

A **continuous** pipeline remains active and aims for lower latency. It has a different cost and operational profile.

Use the least expensive execution model that satisfies the business latency requirement.

## Full Refresh vs Incremental Update

Incremental processing reads new or changed source data. A full refresh recomputes pipeline tables.

Full refreshes are useful after:

- Correcting historical transformation logic
- Changing a schema or business rule
- Rebuilding state after a serious issue

They are also expensive and can increase downstream impact. Production runbooks should state when a full refresh is acceptable.

## Databricks Workflows

Databricks Workflows orchestrates notebooks, Python scripts, SQL tasks, JARs, pipelines, and other jobs.

A multi-task job forms a directed acyclic graph:

```text
ingest_customers ----\
                      -> build_silver -> publish_gold
ingest_orders -------/
```

Important settings include:

- Task dependencies
- Parameters
- Retry policy
- Timeout
- Maximum concurrent runs
- Compute selection
- Notifications
- Schedule or file-arrival trigger

A task should be small enough to retry independently and clear enough that its failure points to a specific stage.

## Job Compute vs Interactive Compute

Interactive compute is convenient for development. Production jobs generally benefit from isolated job compute or serverless job resources:

- Reproducible configuration
- Cleaner dependency isolation
- Automatic lifecycle management
- Reduced risk of interference from interactive users

Cluster pools can reduce startup time for classic compute by keeping instances ready. Serverless options can remove much of the infrastructure management where supported.

## Databricks SQL Warehouses

A SQL warehouse provides compute for Databricks SQL queries, dashboards, and alerts. Its configuration affects concurrency, startup latency, and cost.

Key controls include:

- Warehouse size
- Scaling limits
- Auto stop
- Serverless or classic mode
- Permissions
- Query history and monitoring

Scale up for demanding individual queries. Scale out when many queries must run concurrently.

## Dashboards and Alerts

Saved queries can feed dashboards. A dashboard should present metrics with clear ownership and refresh expectations rather than simply collecting every available chart.

Alerts evaluate query results against a condition:

```sql
SELECT count(*) AS invalid_order_count
FROM orders_silver
WHERE order_id IS NULL;
```

An alert can notify email, Slack, Microsoft Teams, or another configured destination when the result crosses a threshold.

Useful alerts include:

- Pipeline failures
- Freshness delays
- Unexpected row-count changes
- Null or duplicate thresholds
- Data-quality expectation failures
- Excessive query duration

## Production Checklist

1. Keep pipeline code in version control.
2. Parameterize environment-specific paths and catalogs.
3. Use service principals for automation where appropriate.
4. Separate development, test, and production data.
5. Define retries and timeouts.
6. Emit quality and freshness metrics.
7. Configure failure notifications.
8. Test reruns and idempotency.
9. Document full-refresh procedures.
10. Review compute cost after representative runs.

## Source Notes

This article was developed from my Notion notes: [5. Production Pipelines](https://app.notion.com/p/8ab88adf005743758e908eccc0f081c0).

