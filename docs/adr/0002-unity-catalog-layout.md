# ADR-0002: Unity Catalog layout per environment

- Status: Proposed
- Date: 2026-10-28

## Context

Today every table is an external Delta table registered in a single catalog and schema (`dbrtfmprod.whale_trades`) on hard-coded ADLS Gen2 paths (`abfss://datalake@statfmprod.dfs.core.windows.net/...`). Raw files are deleted after loading, so bronze cannot be rebuilt, and there is no separation between environments or layers.

[ADR-0001](0001-migrate-to-declarative-automation-bundles.md) moves the project to one bundle per domain (`exchangeinfo`, `whale_trades`), each with `dev`, `staging` and `prod` targets. This needs a storage layout that isolates environments and developers, and in which every schema and volume belongs to exactly one bundle: a schema declared in two bundles would be created, changed and destroyed by both deployments.

Databricks guidance relevant to this decision:

- One Unity Catalog metastore with separate catalogs for dev, staging and production that mirror the workspace layout; personal schemas for developers in non-production catalogs; the production catalog bound in `ISOLATED` mode to the production workspace ([Developer best practices](https://docs.databricks.com/aws/en/developers/best-practices)).
- During development, bundles use a personal schema per user, such as `dev_${user_name}` ([Developer best practices](https://docs.databricks.com/aws/en/developers/best-practices)).
- The medallion layers are stored in separate schemas ([What is the medallion lakehouse architecture?](https://docs.databricks.com/aws/en/lakehouse/medallion)).
- Volumes govern non-tabular files; one of their use cases is registering landing areas for raw data in the early stages of ETL pipelines. External volumes register an existing cloud storage path, and data remains in cloud storage when the volume is dropped ([What are Unity Catalog volumes?](https://docs.databricks.com/aws/en/volumes/)).
- Build artifacts such as wheels are not committed to Git but uploaded to Unity Catalog volumes during CI ([Developer best practices](https://docs.databricks.com/aws/en/developers/best-practices)).

## Decision

1. **One catalog per environment**, named by the `catalog` bundle variable of each target (for example `dev`, `staging`, `prod`). The production catalog is bound in `ISOLATED` mode to the production workspace.
2. **One schema per domain and layer** inside each catalog, named `<domain>_<layer>`:
   - `exchangeinfo`: `exchangeinfo_landing`, `exchangeinfo_bronze`, `exchangeinfo_silver`, `exchangeinfo_gold`.
   - `whale_trades`: `whale_trades_landing`, `whale_trades_bronze`, `whale_trades_silver`, `whale_trades_gold`, `whale_trades_ml` (registered models).

   Table names carry no domain prefix, for example `exchangeinfo_silver.instruments` or `whale_trades_gold.symbol_scores`.
3. **Personal schemas in development.** In the `dev` target every schema name gets the prefix `dev_<user_short_name>_`, set through a bundle variable, so each developer works on `dev.dev_<user>_exchangeinfo_bronze` and so on, without overwriting anyone else.
4. **Landing in a volume, never modified.** Raw API responses are written as NDJSON to the `raw` volume of the domain's landing schema, in one new folder per run: `/Volumes/<catalog>/<domain>_landing/raw/<dataset>/extraction_date=YYYY-MM-DD/run_ts=YYYYMMDDTHHMMSSZ/`. In `staging` and `prod` the volume is external, on a storage path given per target; in `dev` it is managed, so developers need no external location.
5. **Managed tables for every table.** Bronze, silver and gold tables are managed tables created by the Lakeflow pipelines, with no storage paths in the code.
6. **Artifacts volume shared by staging and production.** CI uploads each versioned wheel once to `/Volumes/<artifacts_catalog>/<artifacts_schema>/wheels/`, a volume that both the staging and the production workspaces can read, so production deploys the exact file that passed staging and every bundle of a release uses the same wheel.
7. **Each bundle manages the schemas and volumes of its domain.** Catalogs, external locations, storage credentials, workspace bindings and the artifacts volume are platform resources created by an administrator, outside the bundles.
8. **Reads across domains through grants.** `whale_trades` reads the `exchangeinfo` tables it needs by name, passed as bundle variables (ADR-0001), and its run-as service principal holds `USE SCHEMA` and `SELECT` only on those schemas.

## Consequences

Positive:

- Environments and developers are isolated by catalog and schema, and production data is unreachable from other workspaces.
- Every schema and volume has a single owning bundle, so each domain is deployed, promoted, rolled back and destroyed without touching the other.
- Grants can be given per domain and layer, for example `SELECT` on `exchangeinfo_gold` only.
- Every bronze table can be rebuilt from the landing volume, which keeps every raw file.
- No storage path or environment name is hard-coded: all names come from bundle variables.

Negative:

- More schemas: nine per catalog, and nine per developer in `dev`.
- A grant on a whole layer (for example every gold schema) is one grant per domain.
- The landing volumes grow with every run; a retention policy, defined in the data contracts, is needed to control their size.
- Existing tables in `dbrtfmprod.whale_trades` are not migrated in place: the new layout is populated by the pipelines, and the old tables are dropped once production runs from the bundles.
- An administrator must create the catalogs, the external locations for the landing volumes and the artifacts volume before the first `staging` or `prod` deployment.

## Alternatives considered

1. **Keep external tables on fixed ADLS paths.** Couples the code to one storage account and one environment, and leaves table lifecycle outside Unity Catalog. Rejected.
2. **One schema per layer shared by all domains** (`bronze`, `silver`, `gold`). Matches the medallion example, but each shared schema would have to be declared in both bundles or moved out of the bundles as a platform resource, so the domains could no longer be deployed or destroyed independently. Rejected.
3. **One schema per domain, with every layer inside it.** Gives each bundle a single schema, but mixes raw, validated and business-ready data and makes layer-level grants per table. Rejected.
4. **One catalog per domain.** Gives domain-level isolation that this project, with a single team, does not need, and multiplies catalogs by environments. Rejected.
