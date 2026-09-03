# Changelog

## 0.2.0 (2026-09-03)

- New `mode` input. The default `pr` is unchanged. `mode: report` traces **every** model in the config at the current commit, with no base and no diff against a branch, and upserts one issue titled `Neurarch weekly model report`.
- The report body carries: a table of every model (layers, params, shape, will it run, GPU fit, estimated cost, fingerprint); every blocker and warning across all models, quoted in full; what changed since the last report; and documentation drift.
- **Changed since the last report** reads the previous report's numbers back out of the issue it is about to overwrite. The body ends in a machine-readable block, `<!-- neurarch-report-data {...} -->`, so the comparison is against measured numbers and not against parsed prose.
- **Documentation drift** looks for a `MODEL_CARD.md` or `README.md` next to each model file and reports any parameter count written there that the trace disagrees with by more than 1 percent. A doc that names other listed models and not this one is skipped, so one card shared by a folder does not accuse every model in it; a doc where any number agrees is treated as right and stays silent.
- **Report mode never fails the job.** A report is a record, not a gate: blockers are printed, the run stays green. A report that crashes outright still fails, because a report that did not happen is a failure.
- The issue is upserted: an open issue with that exact title is updated in place and gets a short comment naming what moved, so the record is one issue with a history and not a weekly pile. A new issue gets the `neurarch` label when the repository already has one, and a repository without it is not a failure.
- New outputs `issue-number` and `issue-url`.

## 0.1.0 (2026-09-03)

First release.

- Composite GitHub Action: on every pull request, trace each model listed in `.neurarch.yml` that the PR touches (HEAD and base), POST both graphs to `https://www.neurarch.com/api/v1/plan`, and post one comment per PR with the plan for every affected model: params, cost, GPU fit, whether it will run, and the deltas against the base branch.
- The comment is upserted (marker `<!-- neurarch-bot -->` on the first line), so a PR carries one comment that follows its commits.
- A blocker fails the check; `fail-on-warn` extends that to warnings.
- Static fallback: when a model cannot be traced (import error, bad input, a constructor that raises), the comment carries the traceback tail and a static read of the file by neurarch-lint. A blocker found that way still fails the check; the trace failure by itself does not.
- Outputs: `legal`, `blockers`, `warnings`, `plans-json`.
