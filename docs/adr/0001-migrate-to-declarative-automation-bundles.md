# ADR-0001: Migrate from Azure Data Factory and notebooks to Declarative Automation Bundles with a Python package

- Status: Proposed
- Date: 2026-10-28

## Context

The batch side of the project runs as follows:

- An Azure Data Factory pipeline (`pipelines/pipeline/exchangeinfo_pipeline_futures_and_spot.json`) copies the spot and futures `exchangeInfo` responses of Binance, Bybit and OKX into ADLS Gen2 with six Copy activities, then triggers three Databricks jobs in sequence: `exchangeinfo_futures_and_spot`, `whale_trades-calculate_scores` and `whale_trades-ETL_and_train`.
- The business logic lives in Databricks notebooks (`notebooks/exchangeinfo/`, `notebooks/etl-train/`), with schemas, transformations and configuration inline.
- The Databricks jobs, their tasks, compute and schedules are defined only in the workspace. The repository holds ADF JSON exports, not a deployable definition of the Databricks side.

This leads to the following problems:

- Job definitions are not version controlled, so changes cannot be reviewed, reproduced or rolled back.
- There is a single environment: no isolated development or staging target and no automated promotion.
- Logic inside notebooks cannot be unit tested or reused, and there is no CI.
- Orchestration is split between two services (ADF and Databricks jobs), each with its own configuration, monitoring and credentials.

Databricks guidance relevant to this decision:

- Declarative Automation Bundles are the recommended approach to CI/CD on Databricks ([CI/CD on Databricks](https://docs.databricks.com/aws/en/dev-tools/ci-cd/)). They describe jobs and pipelines as source files and include tests, configuration and per-environment targets ([What are Declarative Automation Bundles?](https://docs.databricks.com/aws/en/dev-tools/bundles/)).
- Everything that defines the project should be version controlled, in a single repository; multiple bundles for separate deployment lifecycles are kept in that same repository ([Developer best practices](https://docs.databricks.com/aws/en/developers/best-practices)).
- Notebooks should not hold business logic: Python goes into importable modules under `src/`, covered by `pytest`, and notebooks remain thin orchestration and visualization layers. Notebook-heavy projects should be migrated incrementally, one module at a time ([Developer best practices](https://docs.databricks.com/aws/en/developers/best-practices)).
- Bundles should be small and focused, each covering all environments. Databricks recommends separate bundles for different products or domains, for workloads with clearly different lifecycles and where independent promotion or rollback is needed. When one bundle depends on assets published by another, the dependency is modelled in CI/CD rather than by collapsing both into one bundle ([Developer best practices](https://docs.databricks.com/aws/en/developers/best-practices)).
- Artifacts should always be versioned, for example with Git commit hashes, for traceability and rollback ([CI/CD workflows on Databricks](https://docs.databricks.com/aws/en/dev-tools/ci-cd/flows)).

## Decision

Replace Azure Data Factory and the workspace-defined jobs with Declarative Automation Bundles and a Python package, migrated domain by domain:

1. **One Python package.** All business logic moves to an importable package under `src/`, covered by unit tests. CI builds it once per commit as a versioned wheel and stores it in a Unity Catalog volume for artifacts. Notebooks are reduced to thin entry points or replaced by Python wheel tasks.
2. **One bundle per domain.** Each domain gets its own bundle in the same repository, with `dev`, `staging` and `prod` targets, so that it can be deployed, promoted and rolled back independently:
   - `exchangeinfo`: ingestion of exchange metadata through the medallion layers.
   - `whale_trades`: market scoring, trade ingestion, feature engineering, labelling and model training (today in `notebooks/etl-train/`).

   Each bundle lives in its own folder under `bundles/` and its tasks reference the exact wheel version built by CI. Code shared by the domains stays in the package, so no bundle copies another bundle's source.
3. **Orchestration in Lakeflow Jobs, transformations in Lakeflow pipelines.** The ADF Copy activities become ingestion tasks in the package that call the exchange APIs and write to the landing area; the medallion layers become Lakeflow pipelines whose datasets call transformation functions from the package; task order, schedules and retries are defined in each bundle. Azure Data Factory is decommissioned once each domain runs from its bundle in production.
4. **Dependencies between bundles in CI/CD.** `whale_trades` consumes the tables published by `exchangeinfo`. CD deploys and validates `exchangeinfo` first and deploys `whale_trades` only after that succeeds; the names of the upstream tables are passed to `whale_trades` as bundle variables, and its jobs fail fast when an upstream table is missing ([Developer best practices](https://docs.databricks.com/aws/en/developers/best-practices)).
5. **CI/CD in GitHub Actions.** Pull requests run lint, unit tests and `databricks bundle validate` for every bundle and target; merges to `main` deploy to `staging`; releases deploy the same versioned wheel to `prod`. CI/CD authenticates as the deployment service principal. Databricks recommends workload identity federation for CI/CD ([CI/CD workflows on Databricks](https://docs.databricks.com/aws/en/dev-tools/ci-cd/flows)), but federation policies are configured at the account level ([Configure a federation policy](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-federation-policy)) and the current platform, Databricks Free Edition, has no account console or account APIs ([Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)). Until it has, CI/CD uses OAuth machine-to-machine authentication with a secret of the deployment service principal, stored as a GitHub environment secret and rotated before it expires.

Out of scope, each to be covered by its own ADR:

- The deployment platform of the real-time inference client in `streaming/` (a WebSocket consumer on Azure Container Instances). Its code moves into the package so that it shares the feature definitions used in training, but where it runs is a separate decision.
- The storage layout in Unity Catalog (catalogs, schemas, volumes and tables per environment), including which bundle owns each schema, decided in [ADR-0002](0002-unity-catalog-layout.md).
- The experimentation notebooks in `notebooks/experiments/`, which are not deployed.

## Consequences

Positive:

- Jobs, pipelines, schedules and permissions are reviewed in pull requests, versioned and reproducible in every environment.
- Development happens in an isolated `dev` target, and changes are tested in `staging` before reaching `prod`.
- Business logic is unit tested, and every production run traces back to a release, a wheel version and a commit.
- Each domain can be promoted or rolled back without redeploying the other.
- Orchestration, monitoring and alerting live in a single service.

Negative:

- The migration is a sizeable effort: every notebook must be refactored into modules with tests, and both domains must be re-verified end to end.
- API ingestion previously handled by ADF Copy activities (retries, pagination) must be implemented and tested in the package.
- CI/CD must validate and deploy several bundles in dependency order, and the shared wheel version must be the same for every bundle of a release.
- Until workload identity federation is available, CI/CD depends on a long-lived service principal secret that must be rotated.
- During the migration both orchestrators coexist for the domains not yet migrated.
- Contributors need the Databricks CLI and `uv` locally.

## Alternatives considered

1. **Keep Azure Data Factory and notebooks, adding ADF Git integration.** Versions the ADF definitions but leaves the Databricks job definitions, the untestable notebook logic and the two-orchestrator split unchanged. Rejected.
2. **Git folders or Git with jobs.** Databricks offers these to source-control code files only; job configurations such as task sequences, compute settings and schedules stay outside source control, which makes them less suitable for multi-environment deployments ([CI/CD on Databricks](https://docs.databricks.com/aws/en/dev-tools/ci-cd/)). Rejected.
3. **Databricks Terraform provider for jobs.** Databricks recommends Terraform only for cloud-level, external and admin resources, and bundles for all other Databricks resources ([Developer best practices](https://docs.databricks.com/aws/en/developers/best-practices)). Rejected for jobs and pipelines; Terraform remains an option for workspace and cloud infrastructure.
4. **A single bundle for the whole project.** Databricks recommends putting everything a single team owns into one bundle, and this project has one team. It also recommends separate bundles for different domains and for independent promotion or rollback, and modelling dependencies between bundles in CI/CD rather than collapsing them into one bundle ([Developer best practices](https://docs.databricks.com/aws/en/developers/best-practices)). A single bundle would couple the release and rollback of both domains. Rejected.
5. **One Python package per domain.** Would let each bundle release its own wheel, but duplicates shared code (HTTP client, landing writer, logging) or needs a third shared package, and multiplies versions to track. Rejected; the single package is split if a domain gains its own owner or release cadence.
