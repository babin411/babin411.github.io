---
title: Data Governance with Databricks Unity Catalog
categories: [Data Engineering, Databricks]
tags: [databricks, unity catalog, governance, privileges, lineage, external locations, security]
author: 'Babin'
pin: false
math: false
image:
  path: /assets/img/databricks-data-governance.png
  alt: Databricks data governance privileges from the course notes
---

Data governance answers four practical questions: what data exists, who can use it, what they can do with it, and where it came from.

Unity Catalog provides a centralized governance layer for Databricks data and AI assets across workspaces. It organizes securable objects, identities, privileges, storage access, discovery, and lineage.

![Databricks governance privileges](/assets/img/databricks-data-governance.png)
_Governance combines identities, securable objects, ownership, and privileges_

## The Three-Level Namespace

Unity Catalog identifies tables and views with:

```text
catalog.schema.object
```

For example:

```sql
SELECT *
FROM production.sales.orders;
```

The hierarchy is:

| Level | Purpose |
|---|---|
| Metastore | Regional governance boundary assigned to workspaces |
| Catalog | Top-level organizational and isolation boundary |
| Schema | Group of tables, views, volumes, and functions |
| Object | The table, view, volume, model, or function being used |

Catalogs often separate environments, business units, or data domains. Schemas organize related objects within a catalog.

## Principals

Privileges are granted to principals:

- **Users** represent people.
- **Service principals** represent applications and automation.
- **Groups** collect users and service principals.

Prefer grants to groups rather than individual users. Group-based access is easier to audit and maintain as teams change.

## Ownership and Privileges

An owner controls a securable object and can normally manage its permissions. Privileges grant narrower capabilities.

Common privileges include:

- `USE CATALOG`
- `USE SCHEMA`
- `SELECT`
- `MODIFY`
- `CREATE TABLE`
- `CREATE VIEW`
- `EXECUTE`
- `READ VOLUME`
- `WRITE VOLUME`

Access is hierarchical. Reading a table usually requires permission to use its parent catalog and schema as well as `SELECT` on the table.

```sql
GRANT USE CATALOG ON CATALOG production TO `data_analysts`;
GRANT USE SCHEMA ON SCHEMA production.sales TO `data_analysts`;
GRANT SELECT ON TABLE production.sales.orders TO `data_analysts`;
```

Review grants with:

```sql
SHOW GRANTS ON TABLE production.sales.orders;
```

## Least Privilege

Least privilege means granting only the access needed for a role.

Examples:

- Analysts receive `SELECT` on curated gold tables.
- Pipeline service principals receive read access to sources and write access to their targets.
- Data stewards manage selected schemas.
- Platform administrators manage metastores and workspace bindings.

Avoid broad grants such as `ALL PRIVILEGES` unless the role genuinely owns the complete lifecycle of the object.

## Storage Credentials and External Locations

Unity Catalog separates cloud authentication from governed paths.

A **storage credential** represents an authentication mechanism for cloud storage. An **external location** combines that credential with a cloud storage path.

This prevents notebooks from embedding cloud secrets and allows administrators to control which paths users can access.

Volumes provide governed file access for non-tabular data:

```text
/Volumes/catalog/schema/volume/
```

They are appropriate for files such as CSV inputs, configuration files, libraries, images, and model artifacts.

## Managed and External Data

Managed tables store data in Unity Catalog-managed locations. External tables reference data whose lifecycle is managed separately.

Use managed tables when Databricks should manage the data lifecycle. Use external tables when the same storage must be shared with other systems or retained independently of the table registration.

Both patterns should still be governed through Unity Catalog.

## Data Discovery and Lineage

Catalog Explorer helps users discover:

- Schemas and tables
- Column definitions and comments
- Owners and permissions
- Sample data where allowed
- Upstream and downstream lineage

Lineage helps answer where a table originated and which notebooks, jobs, dashboards, or tables depend on it. This is especially valuable before changing or deleting a widely used column.

## Views for Controlled Access

A view can expose a limited projection:

```sql
CREATE VIEW production.hr.paris_employees AS
SELECT id, name, city
FROM production.hr.employees
WHERE city = 'Paris';
```

Users can receive access to the view without receiving direct access to every column in the base table, subject to the configured security model.

For sensitive data, also consider row filters, column masks, and dynamic views.

## Governance Checklist

1. Organize catalogs by deliberate boundaries.
2. Grant access to groups.
3. Use service principals for automated pipelines.
4. Apply least privilege at catalog, schema, and object levels.
5. Use external locations and volumes instead of embedded credentials.
6. Set owners to maintained groups where practical.
7. Add comments and classifications to important datasets.
8. Review grants regularly.
9. Use lineage before making breaking changes.
10. Audit access through system tables and account-level controls.

## Source Notes

This article was developed from my Notion notes: [6. Data Governance](https://app.notion.com/p/3578a45f4f2d478f92abd9c7d435332c).

