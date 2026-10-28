# Security Policy

## Supported versions

Only the latest release on `main` receives security fixes.

## Reporting a vulnerability

Do not report security vulnerabilities in public issues, pull requests or discussions.

Report them privately through GitHub: open the repository's **Security** tab and select **Report a vulnerability**. Include a description of the issue, the steps to reproduce it and its potential impact.

You will receive an acknowledgement within 7 days. Once the issue is confirmed, a fix is prepared and released as a patch version, and the advisory is published after the release.

## Secrets

This repository never contains credentials. Secrets are stored in Databricks secret scopes, in the CI/CD secret store or in local environment variables. If you find a credential in this repository or its history, report it as a vulnerability so it can be revoked.
