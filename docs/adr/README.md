# Architecture Decision Records

This folder records the significant design decisions of the project as Architecture Decision Records (ADRs), following the practice described at [adr.github.io](https://adr.github.io/).

## Rules

- One decision per ADR, written before it is implemented and reviewed in a pull request.
- Files are numbered in order: `NNNN-short-title.md`.
- Status is one of `Proposed`, `Accepted` or `Superseded by ADR-NNNN`.
- An accepted ADR is never rewritten. A change of direction is a new ADR that supersedes it.
- Bug fixes and choices inside the scope of an accepted ADR need no new ADR.

## Template

```markdown
# ADR-NNNN: Title

- Status: Proposed
- Date: YYYY-MM-DD

## Context

## Decision

## Consequences

## Alternatives considered
```

## Index

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-migrate-to-declarative-automation-bundles.md) | Migrate from Azure Data Factory and notebooks to Declarative Automation Bundles with a Python package | Proposed |
