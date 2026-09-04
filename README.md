# neurarch-bot

[![CI](https://github.com/neurarch-ai/neurarch-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/neurarch-ai/neurarch-bot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Try Neurarch](https://img.shields.io/badge/Neurarch-try_it-7c3aed)](https://neurarch.com/bot)

**Dependabot for models.** On every pull request that touches a model, neurarch-bot traces the model on your branch and on the base branch, and posts one comment saying what changed: params, cost, GPU fit, and whether the thing will run at all. A blocker turns the check red, so a model that cannot forward-pass never reaches a GPU. With `fix: true` it goes one step further and opens a pull request with the fix, but only one its own gate accepted: see [It opens the fix](#it-opens-the-fix). On a schedule, it keeps one issue up to date with every model you own: see [Weekly report](#weekly-report).

It reads your model the way the runtime does, by importing it and tracing a forward pass, so the numbers are the real ones and not a guess made from the text of the file.

## Install

Add `.github/workflows/neurarch-bot.yml`:

```yaml
name: Neurarch plan
on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  plan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0        # the bot needs the base commit to diff against
      - uses: neurarch-ai/neurarch-bot@v0
```

And `.neurarch.yml` at the repository root:

```yaml
models:
  - path: models/tiny_gpt.py:TinyGPT   # file.py:ClassOrFactory, constructed with no arguments
    input: "1,32:long"                 # batch first; :long for token ids
  - path: models/small_cnn.py:SmallCNN
    input: 1,3,32,32
```

Per model you may also set `extra_inputs` (a list of further `--input` specs, for a forward that takes more than one tensor), `depth` (stop descending at this module depth), `name`, and `policy`.

### House rules: `policy`

A `policy:` block turns your organisation's own lines into blockers and warnings in the same plan, quoted as you wrote them:

```yaml
policy:                      # applies to every model
  max_params: 200M           # blocker
  must_fit: A100-40GB        # blocker
  max_layers: 400            # blocker
  forbid_types: [softmax]    # blocker
  require_types: [layerNorm] # blocker
  max_memory_gb: 32          # warning
  max_cost_usd: 50           # warning

models:
  - path: models/small_cnn.py:SmallCNN
    input: 1,3,32,32
    policy:
      max_params: 500K       # this model's tighter cap
```

A model's own `policy` merges over the top-level one key by key, so tightening one line does not drop the rest. A policy needs no API key: it can only make the verdict stricter, never more permissive. An unreadable line is reported inside the plan rather than failing the run.

A pull request affects a listed model when the listed file changed, or when any other file in its directory changed (a helper module it imports). A sibling that is itself a listed model does not count: editing `models/cnn.py` is a change to the CNN entry, not to every model that shares the folder. With no `.neurarch.yml` the Action prints how to add one and exits 0.

## What the comment looks like

> ### `models/small_cnn.py:SmallCNN` :white_check_mark: will run
>
> **SmallCNN: 21 layers, 971.9K params, will run, fits a T4, about $0.05 to train, vs SmallCNN: params +830.6K**
>
> | | |
> |---|---|
> | Layers / params | 21 / 971.9K (+830.6K, +588%) |
> | Shape | `[3,32,32]` -> `[10]` |
> | Will it run | yes, 0 warnings |
> | GPU fit | T4 yes · A100 40GB yes · H100 80GB yes (memory +0.01 GB) |
> | Est. training | ~$0.05 / 3m on A10G (24GB) (+<$0.01) |
> | Blast radius | 10 downstream layers from 5 touched |

One comment per pull request, carrying one section per affected model. It is upserted on every push, so it follows your commits instead of piling up.

When a model cannot be traced (an import error, a bad `input`, a constructor that raises), that model's section carries the tail of the traceback and a static read of the file by [neurarch-lint](https://github.com/neurarch-ai/neurarch-lint) instead. A failed trace does not by itself fail the job; a blocker found by the static read does.

## It opens the fix

Set `fix: true` and a blocker stops being a message. The bot patches the model, checks its own work, and opens a pull request:

```yaml
jobs:
  plan:
    runs-on: ubuntu-latest
    permissions:
      contents: write        # push the fix branch
      pull-requests: write   # comment, and open the fix pull request
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: neurarch-ai/neurarch-bot@v0
        with:
          fix: true
          api-key: ${{ secrets.NEURARCH_API_KEY }}   # optional; see below
```

It runs **after** the plan comment is posted, because the plan is the product and it reaches the pull request whether or not a fix turns out to be possible.

**Two sources of patches, in order.**

1. **The exact patch**, from `suggest_fix` in [neurarch-mcp](https://github.com/neurarch-ai/neurarch-mcp). It reads the `.py`, lints it, and returns a unified diff for the rules that pin a number or an order. No model call, no key, nothing metered. There is no one-shot `suggest_fix` subcommand on that CLI, so the bot speaks the server's stdio JSON-RPC directly: two writes, two reads, standard library, no MCP client library on the runner. Node is already on the GitHub runners; a runner without it just skips this source and says so.
2. **A proposal**, from `POST /api/v1/fix`. This one costs: it is metered against the API key owner's agent quota. With no `api-key` input it is **skipped outright** and the comment says the source was skipped rather than pretending it was tried.

**Nothing is pushed that the bot's own gate has not accepted.** The patch goes into a scratch worktree at the pull request's head, the model is traced again and planned again, and the result is accepted only when:

- the new plan is **legal** (no blockers, policy lines included), and
- the parameter count is **within 10 percent** of the pull request's own head. A "fix" that deletes half the model turns every blocker green and is not a fix.

Up to **3 rounds**, feeding the findings that are left back into the sources. A round that reduces the blocker count is kept and built on; a round that does not is thrown away rather than left to poison the next one. A patch against any file `.neurarch.yml` does not list is refused before it is applied.

On success the branch `neurarch/fix-pr<N>` is pushed and a pull request titled `Fix the blocker in #<N>` is opened **into the original pull request's head branch**, not into the default branch: the fix belongs on top of the change that broke it. Its body carries both plans side by side and names which source produced the patch. One line is added to the plan comment linking it.

On failure nothing is pushed, and one line is added to the plan comment saying what was tried and why it was rejected.

**A second fix pull request is never opened.** The branch name is derived from the pull request number, so a re-run pushes the same branch and updates the pull request that is already open on it.

### The fix pull request has no check of its own

A pull request opened with the default `GITHUB_TOKEN` **does not trigger workflows**. That is GitHub's rule, not ours, and it is there so an Action cannot start itself in a loop. So the fix pull request arrives without its own `Neurarch plan` check, even though the bot verified the patch before pushing it.

If you want the fix pull request to be checked like any other, pass a personal access token (`contents: write`, `pull-requests: write`) as `github-token`:

```yaml
      - uses: neurarch-ai/neurarch-bot@v0
        with:
          fix: true
          github-token: ${{ secrets.NEURARCH_BOT_PAT }}
```

## Weekly report

A pull request only ever shows you the model somebody touched. `mode: report` traces **every** model in `.neurarch.yml` at the current commit and keeps one issue up to date with what you own: what each model costs, whether it still runs, what moved since last week, and where the docs have drifted away from the code.

Add `.github/workflows/neurarch-report.yml`:

```yaml
name: Neurarch weekly report
on:
  schedule:
    - cron: '0 13 * * 1'   # Mondays, 13:00 UTC
  workflow_dispatch:        # and on demand

permissions:
  contents: read
  issues: write             # the report is an issue, upserted every run

jobs:
  report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: neurarch-ai/neurarch-bot@v0
        with:
          mode: report
```

No `fetch-depth: 0` here: report mode traces one commit and never looks at a base branch.

### What the issue looks like

> ## Neurarch weekly model report
>
> Every model listed in `.neurarch.yml`, traced at [`62290e1`](#) on 2026-09-03. This is a record, not a gate: report mode never fails the job.
>
> | Model | Layers | Params | Shape | Will it run | GPU fit | Est. cost | Fingerprint |
> |---|---|---|---|---|---|---|---|
> | `models/tiny_gpt.py:TinyGPT` | 42 | 13.5M | `[32]` -> `[32,8192]` | :warning: yes (1 warning) | A10G (24GB) | ~$0.05 | `da073440` |
> | `models/small_cnn.py:SmallCNN` | 21 | 141.4K | `[3,32,32]` -> `[10]` | :white_check_mark: yes | A10G (24GB) | ~$0.04 | `0a82eb9e` |
>
> ### Blockers and warnings
>
> 0 blockers and 1 warning across 2 models.
>
> #### `models/tiny_gpt.py:TinyGPT`
>
> - :warning: warning: **Attention without positional encoding** - 4 attention layer(s) present but no positional encoding found.
>   - Fix: Add a PositionalEncoding (sinusoidal) or RoPE layer before the first attention layer.
>
> ### Changed since the last report
>
> Against the report of 2026-08-25:
>
> - `models/small_cnn.py:SmallCNN`: params 971.9K -> 141.4K (-85%); layers 25 -> 21; fingerprint `deadbeef` -> `0a82eb9e`
>
> ### Documentation drift
>
> - `models/small_cnn.py:SmallCNN`: `models/MODEL_CARD.md` says 1.2M ("Parameters | 1,200,000"), the trace says 141.4K (+749%)

Live example: [neurarch-ai/model-ci-example](https://github.com/neurarch-ai/model-ci-example/issues?q=is%3Aissue+Neurarch+weekly+model+report).

Four things worth knowing:

- **One issue, not one a week.** The Action finds the open issue with that exact title, rewrites the body, and adds a short comment naming what moved. You get one place to look and a history of changes underneath it. If the repository has a `neurarch` label, a newly created issue gets it.
- **The diff is against measured numbers.** The body ends in `<!-- neurarch-report-data {...} -->`, a compact JSON record of what this run traced. Next week's run reads it back out of the issue it is about to overwrite, so "params 971.9K -> 141.4K" comes from two traces and not from parsing prose. Delete the block by hand and the next report simply says it is the first one.
- **Documentation drift** reads a `MODEL_CARD.md` or `README.md` sitting next to the model file and compares any parameter count written there against the traced one, flagging a gap over 1 percent. A doc that names other listed models but not this one is skipped, so a card shared by a folder does not accuse every model in it, and a doc where some number already agrees is treated as right. No number written anywhere means nothing is reported.
- **It never fails the job.** A report is a record. Blockers are printed in full and the run stays green; keep `mode: pr` for the gate. A report that crashes outright does fail, because a report that did not happen is a failure.

## Inputs

| Input | Default | What it does |
|---|---|---|
| `mode` | `pr` | `pr` plans the models a pull request touches and comments on it. `report` traces every listed model and upserts the weekly issue. |
| `config` | `.neurarch.yml` | Path to the config listing the models, relative to the repository root. |
| `api-key` | *(none)* | A Neurarch API key (`nrk_...`). Optional; keyless calls are rate-limited per hour and per IP. Also unlocks the server fix source, which is metered against the key owner's agent quota. |
| `fix` | `false` | On a blocker, patch the model, re-trace, re-plan, and open a fix pull request if the new plan is legal and the parameter count is within 10 percent of the head. Needs `contents: write`. |
| `github-token` | `${{ github.token }}` | Token used to push the fix branch, open the fix pull request and add one line to the plan comment. A pull request opened with the default token gets no workflow run; pass a PAT to get one. |
| `fail-on-warn` | `false` | Fail the check on warnings, not just blockers. Ignored in report mode, which never fails on findings. |
| `comment-on-clean` | `false` | Post a comment even when the traced structure is identical to the base and has no findings. An existing comment is updated either way. |
| `python-version` | `3.11` | Python used to import the model and run the trace. |
| `torch-index-url` | `https://download.pytorch.org/whl/cpu` | Wheel index for torch. The default is the CPU build, which is all a trace needs. |

## Outputs

| Output | What it carries |
|---|---|
| `legal` | `true` when every affected model will run. |
| `blockers` | Total blockers across affected models. |
| `warnings` | Total warnings across affected models. |
| `plans-json` | JSON array, one row per affected model: target, input, status, legal, blockers, warnings, summary, params, cost, and the deltas when a base was traced. |
| `fix-status` | `none`, `not-fixed`, `opened`, `updated`, `push-failed` or `pr-failed`. |
| `fix-pr-number` | The number of the fix pull request, when one was opened or updated. |
| `fix-pr-url` | The URL of the fix pull request. |
| `issue-number` | In report mode, the number of the weekly report issue created or updated. |
| `issue-url` | In report mode, the URL of that issue. |

## Privacy

The traced graph leaves your repository: layer structure and shapes only, never weights and never data. Nothing is published unless you ask.

Your **source files** are uploaded only by the server fix source, and only when you set both `fix: true` and an `api-key`: `POST /api/v1/fix` receives the model files `.neurarch.yml` names, because a patch cannot be written against a file nobody has read. It stores nothing and runs nothing. The exact patch source reads your files on the runner and sends nothing anywhere. See [neurarch.com/bot](https://neurarch.com/bot).

## License

MIT. See [LICENSE](./LICENSE).
