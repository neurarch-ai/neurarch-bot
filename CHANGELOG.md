# Changelog

## 0.3.0 (2026-09-03)

- **New `fix` input.** On a blocker, the bot patches the model, verifies its own work, and opens a pull request. It runs after the plan comment is posted, because the plan reaches the pull request whether or not a fix turns out to be possible.
- **Two sources of patches, in order.** The **exact patch** comes from `suggest_fix` in `neurarch-mcp`: it reads the `.py`, lints it and returns a unified diff for the rules that pin a number or an order, at no model call and no key. That CLI has no one-shot `suggest_fix` subcommand, so the bot speaks the server's stdio JSON-RPC directly (two writes, two reads, standard library, no MCP client library on the runner). The **proposal** comes from `POST /api/v1/fix`, which is metered against the API key owner's agent quota and is therefore **skipped outright when no `api-key` is set**, with the comment saying the source was skipped rather than pretending it was tried.
- **Nothing is pushed that the bot's own gate has not accepted.** The patch is applied to a scratch worktree at the pull request's head, the model is traced again and planned again, and the result is accepted only if the new plan is legal AND the parameter count is within 10 percent of that head. A "fix" that deletes half the model turns every blocker green and is not a fix. Up to 3 rounds, feeding the remaining findings back in; a round that reduces the blocker count is kept and built on, a round that does not is thrown away.
- **The fix pull request goes into the original pull request's head branch**, not the default branch: the fix belongs on top of the change that broke it. Branch `neurarch/fix-pr<N>`, title `Fix the blocker in #<N>`, body carrying both plans side by side and naming which source produced the patch. One line is added to the plan comment linking it; on failure, one line saying what was tried and why it was rejected.
- **Never a second fix pull request for the same original.** The branch name is derived from the pull request number, so a re-run pushes the same branch and updates the pull request already open on it.
- **A patch is refused before it is applied** if it targets a file `.neurarch.yml` does not list, is absolute, or climbs out of the repository. The bot has a write token; the allowlist is the config's own model files and nothing else.
- New `github-token` input, defaulting to `${{ github.token }}`. A pull request opened with the default token **does not trigger workflows**, so the fix pull request arrives with no bot check of its own; pass a PAT to get one.
- **New `policy:` block in `.neurarch.yml`**, sent with every plan: `max_params`, `must_fit`, `max_layers`, `forbid_types` and `require_types` are blockers, `max_memory_gb` and `max_cost_usd` are warnings, each quoted as you wrote it. A model's own `policy` merges over the top-level one key by key, so tightening one line does not drop the rest. It needs no API key, because a policy can only make the verdict stricter.
- **One repository setting is needed, or the pull request 403s.** GitHub blocks an Action from opening a pull request unless "Allow GitHub Actions to create and approve pull requests" is on. The branch pushes, the API call comes back 403, and the bot says it pushed the branch but could not open the pull request. The README says where the toggle is.
- New outputs `fix-status`, `fix-pr-number`, `fix-pr-url`.

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
