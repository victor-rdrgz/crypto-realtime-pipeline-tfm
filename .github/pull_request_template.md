## Summary

<!-- The problem, why it needs solving, the approach and the result. -->

## Where to look first

<!-- The file or change the reviewer should start with. -->

## Linked issue

<!-- "Closes #<issue>" when this pull request targets main; "#<issue>" otherwise. Remove for trivial changes. -->

## Verification

<!-- How the change was verified: tests, bundle validate, the dev deployment and job run, the data written and the logged metrics. -->

## Definition of Done

- [ ] Tests added or updated, and passing locally
- [ ] Lint and format checks pass
- [ ] `databricks bundle validate` passes for every target (when the bundle changes)
- [ ] Verified in `dev` (deployment and job run), with the result described above
- [ ] ADR, documentation, runbook and data contracts updated where needed
- [ ] No secrets and no AI assistant files in the diff
- [ ] Title follows Conventional Commits and a label is set
