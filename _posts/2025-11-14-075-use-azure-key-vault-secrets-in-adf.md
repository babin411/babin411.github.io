---
title: "Use Azure Key Vault Secrets in ADF"
categories: [Cloud, Azure, Azure Data Factory]
tags: [azure, azure data factory, adf, data engineering]
author: 'Babin'
pin: false
math: false
image:
  path: /assets/img/adf-fundamentals.svg
  alt: Azure Data Factory fundamentals
---

Use Azure Key Vault Secrets in ADF focuses on safer secret management. Production ADF pipelines should avoid hard-coded secrets and pull credentials from a controlled secret store.

This post is part of my Azure Data Factory tutorial notes. The goal is to turn the lesson into a practical blog reference: what the feature does, where it fits in an ADF project, what to configure, and what to check before relying on it.

![Azure Data Factory fundamentals](/assets/img/adf-fundamentals.svg)
_Azure Data Factory fundamentals_

## Where This Fits

Azure Data Factory is an orchestration and data integration service. A typical ADF solution uses linked services for connections, datasets for data shape and location, pipelines for orchestration, activities for work, triggers for scheduling or events, and monitoring for operational visibility.

**Use Azure Key Vault Secrets in ADF** fits into that model as a focused building block. It should be understood not only as a screen in the Azure portal, but as a design decision inside the larger pipeline lifecycle.

## Key Ideas

- Store secrets in Azure Key Vault.
- Grant ADF managed identity only the permissions it needs.
- Reference secrets from linked services instead of typing credentials directly.
- Rotate credentials without editing every pipeline.

## Practical Walkthrough

Start with a small factory or development environment. Keep the first version narrow: one source, one destination, one activity chain, and a clear success condition. This makes the behavior of Use Azure Key Vault Secrets in ADF easier to see before it is hidden inside a larger production workflow.

Create a Key Vault secret, grant the factory managed identity access, reference the secret from a linked service, and test the connection without exposing the secret value.

After the first run succeeds, inspect the run details. ADF usually gives useful output such as status, duration, input settings, output JSON, error messages, and integration runtime information. That output is often the fastest way to understand whether the feature is configured correctly.

## Design Notes

ADF projects become hard to maintain when every value is typed directly into every activity. Use parameters for values that change, keep naming consistent, and avoid duplicating connection information. For production work, separate environments, avoid hard-coded secrets, and keep a clear path from development to deployment.

When this feature interacts with files or external systems, also think about retry behavior, partial failure, idempotency, and cleanup. A pipeline that works once in a demo can still fail in production if reruns create duplicate files, overwrite the wrong folder, or reuse stale activity output.

## Validation Checklist

- The pipeline or data flow has a clear purpose and readable activity names.
- Connections, datasets, and parameters are tested with realistic values.
- Monitoring output shows the expected rows, files, branches, or status.
- Failure behavior is understood before the workflow is scheduled.
- Secrets and environment-specific values are not hard-coded.

## Source

Based on my Notion lesson page: [75. Use Azure Key Vault Secrets in ADF](https://app.notion.com/p/2a06da0217054702a527bda9b90e5322).
