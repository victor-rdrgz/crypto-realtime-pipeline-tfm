# Contributing

Thank you for your interest in this project. This guide describes how changes are planned, built, reviewed and released.

## Planning work

- Planned work (a feature, a new data source, a bug or a design question) starts as an issue created from one of the [issue forms](.github/ISSUE_TEMPLATE). Blank issues are disabled.
- Each issue states its purpose and acceptance criteria. A bug report also includes steps to reproduce, the expected result and the actual result.
- Larger work is split into sub-issues, and milestones group the issues of a release.
- Trivial changes (a typo, a broken link, an obvious one-line fix) can go straight to a pull request without an issue.
- Design decisions are recorded as Architecture Decision Records (ADRs) in [`docs/adr/`](docs/adr) and reviewed in a pull request before they are implemented. An accepted ADR is never rewritten; it is superseded by a new one.

## Branches and commits

This project follows [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow).

- Create a short-lived branch from `main` named `<type>/<description>`, for example `feat/okx-ingestion` or `fix/null-symbol`.
- Keep one branch per set of related changes and each commit a complete, isolated change.
- Never commit secrets (API keys, tokens, passwords, connection strings) or local configuration such as `.env` files.

## Pull requests

- Open the pull request as a draft early to get feedback while the work is in progress.
- Keep it small and focused on one purpose; split large changes into several pull requests.
- Use a [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) title, for example `feat: ingest OKX funding rates`. Pull requests are squash-merged and the title becomes the commit message on `main`.
- Fill in the [pull request template](.github/pull_request_template.md): the problem, the approach, the result and where the reviewer should look first.
- Link the issue with a closing keyword (`Closes #123`) when the pull request targets `main`.
- Add a label (`bug`, `enhancement`, `documentation`, ...): release notes are generated from labels.
- Mark the pull request as ready for review once the Definition of Done in the template is complete and all checks pass.

## Releases

Versions follow [Semantic Versioning](https://semver.org/). A release is a tag `vX.Y.Z` on `main` with a GitHub Release whose notes are generated from the merged pull requests, grouped by label according to [`.github/release.yml`](.github/release.yml). There is no hand-written changelog.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Security issues are reported as described in the [Security Policy](SECURITY.md), never in public issues.
