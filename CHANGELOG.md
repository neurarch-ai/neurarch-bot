# Changelog

## 0.1.0 (2026-09-03)

First release.

- Composite GitHub Action: on every pull request, trace each model listed in `.neurarch.yml` that the PR touches (HEAD and base), POST both graphs to `https://www.neurarch.com/api/v1/plan`, and post one comment per PR with the plan for every affected model: params, cost, GPU fit, whether it will run, and the deltas against the base branch.
- The comment is upserted (marker `<!-- neurarch-bot -->` on the first line), so a PR carries one comment that follows its commits.
- A blocker fails the check; `fail-on-warn` extends that to warnings.
- Static fallback: when a model cannot be traced (import error, bad input, a constructor that raises), the comment carries the traceback tail and a static read of the file by neurarch-lint. A blocker found that way still fails the check; the trace failure by itself does not.
- Outputs: `legal`, `blockers`, `warnings`, `plans-json`.
