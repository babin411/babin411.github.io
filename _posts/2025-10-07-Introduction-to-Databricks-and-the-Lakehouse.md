---
title: Introduction to Databricks, Lakehouse Architecture, and Compute
categories: [Data Engineering, Databricks]
tags: [databricks, lakehouse, apache spark, compute, dbfs, photon, data engineering]
author: 'Babin'
pin: false
math: false
image:
  path: /assets/img/databricks-introduction.png
  alt: Databricks lakehouse overview from the course notes
---

Databricks is a multi-cloud data and AI platform built around Apache Spark and the lakehouse architecture. It brings data engineering, analytics, machine learning, and governance into one environment while using cloud object storage as the durable foundation.

This introduction explains the lakehouse idea, the control and data planes, Databricks Runtime, workspaces, compute, DBFS, Photon, and the practical decisions involved in creating a development cluster.

![Databricks lakehouse overview](/assets/img/databricks-introduction.png)
_The Databricks lakehouse combines data-lake flexibility with data-warehouse reliability_

## Why the Lakehouse Exists

Traditional data lakes are inexpensive and flexible, but raw files alone do not provide transactions, reliable schema enforcement, or consistently fast SQL performance. Data warehouses provide stronger management and query performance, but they can be less flexible for unstructured data and machine-learning workloads.

A lakehouse aims to combine both:

- Open cloud storage and flexible data formats
- ACID transactions and reliable table semantics
- Batch and streaming processing
- SQL analytics and business intelligence
- Data science and machine learning
- Centralized security and governance

Databricks implements this model with technologies such as Apache Spark, Delta Lake, Photon, Unity Catalog, and managed workflow services.

## The Main Platform Layers

The platform can be understood as three broad layers:

| Layer | Responsibility |
|---|---|
| Cloud infrastructure | Virtual machines, networking, and object storage from AWS, Azure, or Google Cloud |
| Databricks Runtime | Apache Spark, Delta Lake, system libraries, and optional performance engines |
| Workspace | Notebooks, SQL, jobs, pipelines, repositories, dashboards, and administration |

The workspace is where engineers interact with the platform. The underlying compute and storage still come from the selected cloud provider.

## Control Plane and Compute Plane

Databricks separates platform management from workload execution.

The **control plane** hosts services used to manage the workspace, including the web interface, notebooks, job orchestration, and compute management. The **compute plane** is where clusters or serverless resources process data.

This separation matters because it explains where data is stored and where code runs. Exact deployment details vary by cloud and workspace configuration, so security reviews should use the architecture documentation for the specific environment.

## Apache Spark in Databricks

Spark distributes processing across a cluster:

- The **driver** plans work, coordinates tasks, and collects results.
- **Workers** execute tasks against partitions of data.
- A **single-node** configuration runs work on the driver without separate workers.

Spark supports SQL, Python, Scala, R, and Java. Databricks notebooks can mix supported languages using notebook magic commands, which makes it possible to prepare data with SQL and continue processing it with PySpark.

## Databricks Runtime and Photon

Databricks Runtime is the software image installed on compute. A runtime version determines the bundled Spark version, Delta Lake capabilities, Python and Scala versions, and included libraries.

For production workloads:

- Prefer a supported long-term support runtime when stability matters.
- Test upgrades before changing shared workloads.
- Pin external library versions.
- Record runtime requirements with the project.

**Photon** is Databricks' vectorized execution engine. It can accelerate supported SQL and DataFrame workloads without requiring a rewrite of the query logic.

> The runtime version shown in older screenshots may no longer be appropriate. Choose a currently supported runtime rather than copying the version from a historical lab.
{: .prompt-warning }

## DBFS and Cloud Storage

DBFS provides file-system-style access to storage used by Databricks. It is important to distinguish the interface from the durable storage underneath it.

In modern projects, prefer governed cloud storage, Unity Catalog volumes, and external locations for business data. Avoid designing new production pipelines around legacy DBFS root or mount patterns when a governed storage option is available.

The durable design principle is simple: clusters are replaceable; important data should live in persistent cloud storage.

## Creating Compute for Development

A basic development configuration normally requires decisions about:

1. **Workload type**: interactive development or scheduled job.
2. **Access mode**: selected according to governance and workload requirements.
3. **Runtime**: a supported version compatible with the code.
4. **Node type**: sized for CPU, memory, and workload characteristics.
5. **Scaling**: fixed workers or autoscaling.
6. **Auto termination**: stops idle interactive compute.
7. **Policy**: constrains expensive or insecure configurations.

A single-node resource can be enough for learning and small experiments. Distributed compute is useful when the dataset or transformation benefits from parallelism.

## DBUs and Cost Awareness

A Databricks Unit, or DBU, represents a unit of processing capability used for pricing. Total cost usually combines Databricks usage with the cloud provider's compute, storage, and networking charges.

Cost controls should include:

- Auto termination for interactive compute
- Job compute for scheduled production tasks
- Cluster policies
- Sensible worker limits
- Serverless resources where they fit the workload
- Monitoring for idle or oversized compute

## What to Remember

The most important ideas are:

- Databricks is a lakehouse platform, not just a managed Spark cluster.
- Cloud object storage is the durable data layer.
- The control plane manages the service while compute resources execute workloads.
- Databricks Runtime packages Spark, Delta Lake, and supporting libraries.
- Driver and worker roles explain distributed execution.
- Compute configuration affects performance, security, and cost.
- Modern storage designs should favor Unity Catalog-governed locations.

These fundamentals provide the vocabulary needed for Delta Lake, Spark SQL, streaming, production pipelines, and governance.

## Source Notes

This article was developed from my Notion notes: [1. Introduction](https://app.notion.com/p/1a76b346998b42c0a5d8593cc76f14bb).

