# neurarch-bot

[![CI](https://github.com/neurarch-ai/neurarch-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/neurarch-ai/neurarch-bot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Try Neurarch](https://img.shields.io/badge/Neurarch-try_it-7c3aed)](https://neurarch.com/bot)

**Dependabot for models.** On every pull request that touches a model, neurarch-bot traces the model on your branch and on the base branch, and posts one comment saying what changed: params, cost, GPU fit, and whether the thing will run at all. A blocker turns the check red, so a model that cannot forward-pass never reaches a GPU. With `fix: true` it goes one step further and opens a pull request with the fix, but only one its own gate accepted: see [It opens the fix](#it-opens-the-fix). Comment `/neurarch train` on the pull request and it runs the model on a real GPU and posts the curve back: see [`/neurarch train`](#neurarch-train). On a schedule, it keeps one issue up to date with every model you own: see [Weekly report](#weekly-report).

It reads your model the way the runtime does, by importing it and tracing a forward pass, so the numbers are the real ones and not a guess made from the text of the file.

The rules it enforces are **yours**. A `policy:` block in `.neurarch.yml` turns your team's own lines (a parameter cap, a card everything must still fit, a layer type nobody may use) into blockers on every pull request, and the comment names the line of your config each one came from. You do not have to write that file: [`mode: init`](#mode-init-the-config-writes-itself) walks the repository, traces what it finds, and opens a pull request with a config whose numbers came from the measurement.

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

And `.neurarch.yml` at the repository root. **You do not have to write it**: run the Action once with [`mode: init`](#mode-init-the-config-writes-itself) and it walks the repository, traces every model it can construct, and opens a pull request adding the file with your models in it and house rules derived from what it measured. By hand it looks like this:

```yaml
models:
  - path: models/tiny_gpt.py:TinyGPT   # file.py:ClassOrFactory, constructed with no arguments
    input: "1,32:long"                 # batch first; :long for token ids
  - path: models/small_cnn.py:SmallCNN
    input: 1,3,32,32
```

Per model you may also set `extra_inputs` (a list of further `--input` specs, for a forward that takes more than one tensor), `depth` (stop descending at this module depth), `name`, and `policy`.

[**`examples/.neurarch.yml`**](./examples/.neurarch.yml) is every supported key, annotated: what each one blocks, what each one warns on, and a worked example of each. Copy it and delete what you do not want.

### `mode: init`: the config writes itself

```yaml
name: Neurarch init
on: workflow_dispatch          # run it once, from the Actions tab

permissions:
  contents: write              # to push the branch the config arrives on
  pull-requests: write         # to open the pull request

jobs:
  init:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: neurarch-ai/neurarch-bot@v0
        with:
          mode: init
```

It walks the repository for `nn.Module` subclasses and `*.neurarch.json` graphs (skipping tests, build directories and virtualenvs), guesses each model's input shape from the first layer it constructs, traces it, and writes what came back. Then it opens **one pull request** titled `Add a Neurarch config` whose body explains the file line by line.

Three things it will not do:

- **It never invents a number.** `max_params` is a round number just above the largest model it actually measured here, `must_fit` is the smallest card every measured model actually fits, and the pull request body labels each line **measured** or **a suggestion**. `forbid_types` is the only suggestion, and it arrives commented out.
- **It never commits to a branch anybody works on.** The config arrives as a pull request into the branch the run was on, because a config is a set of decisions and those are yours to make.
- **It does nothing at all if a config already exists.** Not a diff, not a comment. Delete the file first if you want it rewritten.

A model it could not construct (a constructor that takes arguments, an input shape the guess got wrong) is written into the file **commented out with the reason**, so nothing it found is silently dropped. So is a class another class in the same file constructs, which is a building block rather than a model.

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

Every key, annotated, with what it blocks and what it warns on: [`examples/.neurarch.yml`](./examples/.neurarch.yml).

#### The comment names the line the rule is written on

A policy blocker used to read `policy: max_params 500K exceeded (1.1M)` and stop there, which leaves whoever thinks the rule is wrong with nowhere to go. Under the findings, the comment now carries one line per violated rule:

> **The rules that fired, and where they live**
>
> - `max_params: 500K` is at [`.neurarch.yml:17`](#), in this model's own `policy:` block.
> - `forbid_types: [softmax]` is at [`.neurarch.yml:9`](#), in the top-level `policy:` block.
>
> _Each of these is a decision someone here made, not a Neurarch default. If a rule is wrong, that line is the thing to change._

The link goes to the line at the pull request's own commit. A rule the model overrode is quoted from the model's block and not from the top-level one, because that is the order the two are merged in before the verdict is asked for; a rule with no line in the file (it came from somewhere this run did not read) says so rather than pointing at a plausible wrong line.

A pull request affects a listed model when the listed file changed, or when any other file in its directory changed (a helper module it imports). A sibling that is itself a listed model does not count: editing `models/cnn.py` is a change to the CNN entry, not to every model that shares the folder. With no `.neurarch.yml` the Action prints how to add one (or run `mode: init`) and exits 0.

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

### One repository setting, or the pull request 403s

GitHub blocks an Action from opening a pull request unless the repository (or the organisation) allows it. The branch pushes fine, the API call comes back `403 Forbidden`, and the bot reports that it pushed the branch but could not open the pull request. Turn it on once:

**Settings -> Actions -> General -> Workflow permissions -> "Allow GitHub Actions to create and approve pull requests"**

or, from the CLI:

```bash
gh api -X PUT /repos/<owner>/<repo>/actions/permissions/workflow \
  -F default_workflow_permissions=read -F can_approve_pull_request_reviews=true
```

An organisation can also set this for every repository at once, and an organisation policy of "off" wins over the repository's own setting. A `github-token` that is a PAT rather than the default `GITHUB_TOKEN` is not subject to it.

### The fix pull request has no check of its own

A pull request opened with the default `GITHUB_TOKEN` **does not trigger workflows**. That is GitHub's rule, not ours, and it is there so an Action cannot start itself in a loop. So the fix pull request arrives without its own `Neurarch plan` check, even though the bot verified the patch before pushing it.

If you want the fix pull request to be checked like any other, pass a personal access token (`contents: write`, `pull-requests: write`) as `github-token`:

```yaml
      - uses: neurarch-ai/neurarch-bot@v0
        with:
          fix: true
          github-token: ${{ secrets.NEURARCH_BOT_PAT }}
```

## `/neurarch train`

The plan says a model **will** run and what it **would** cost. Comment `/neurarch train` on the pull request and it runs, on a real GPU, and the bot edits one comment as the epochs arrive.

Add the job to the same workflow. It needs `issue_comment` on the trigger list, and a checkout of the pull request's own head, which a comment event does not give you by default:

```yaml
name: Neurarch
on:
  pull_request:
  issue_comment:
    types: [created]

permissions:
  contents: read
  pull-requests: write     # the bot posts and edits its own comment

jobs:
  plan:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: neurarch-ai/neurarch-bot@v0
        with:
          api-key: ${{ secrets.NEURARCH_API_KEY }}

  train:
    if: github.event_name == 'issue_comment' && github.event.issue.pull_request != null && startsWith(github.event.comment.body, '/neurarch train')
    runs-on: ubuntu-latest
    steps:
      # An issue_comment event checks out the default branch, which is not the model anyone
      # asked about. This is the merge commit of the pull request the comment is on.
      - uses: actions/checkout@v4
        with:
          ref: refs/pull/${{ github.event.issue.number }}/merge
      - uses: neurarch-ai/neurarch-bot@v0
        with:
          mode: train
          api-key: ${{ secrets.NEURARCH_API_KEY }}
```

`mode: train` is selected automatically on an `issue_comment` event, so it can be left out; it is written above because a workflow that says what it does is easier to read than one that does not.

### The arguments

```
/neurarch train
/neurarch train epochs=3 dataset=mnist
/neurarch train model=models/small_cnn.py:SmallCNN dataset=cifar10 gpu=T4 batch=128 lr=3e-4 optimizer=sgd
```

| Argument | Default | Notes |
|---|---|---|
| `model` | the only model in `.neurarch.yml` | `file.py:ClassOrFactory`, or just the class name. Required when the config lists more than one, because picking one for you would bill you for a model you did not ask about. |
| `dataset` | `mnist` | `mnist`, `cifar10` or `tabular`. The rest need a signed upload URL, which a comment has no way to obtain. |
| `epochs` | 3 | Clamped to 1..20 by the endpoint; the comment says so when it was. |
| `batch` | 64 | Clamped to 8..512. |
| `lr` | 0.001 | Clamped to 1e-5..1. |
| `optimizer` | `adamw` | `adam`, `adamw` or `sgd`. |
| `gpu` | the endpoint's choice | Any card your key is allowed to ask for. |

Anything the parser does not recognise is **refused, not ignored**: `epoch=3` quietly training the default number of epochs is exactly the failure this rule exists to prevent.

### The order it does things in, which is the point

1. **Write access, or nothing.** A comment is something anyone with a GitHub account can leave, and the key belongs to whoever installed the Action. Only `OWNER`, `MEMBER` and `COLLABORATOR` can spend it; everyone else gets one sentence saying so, and nothing is submitted.
2. **A key, or nothing.** There is no keyless training.
3. **Your house rules, then the verifier.** `POST /api/v1/train` plans the graph on its own and never sees your `policy:` block, so the bot asks `/api/v1/plan` for the verdict first, for free, and submits only what your own rules allow. A `max_params` line that fails after the GPU is booked has saved nobody anything.
4. **A blocked design is refused before anything is booked.** The comment carries the blockers and says, in those words, that nothing was submitted and nothing was spent. That refusal is the product working: generating an architecture is cheap, and training one that cannot forward-pass is not.
5. **No GPU, no simulation.** If managed training is not configured for your key the endpoint answers 503 and the bot says so. It will never post a simulated curve into a pull request as a training result.
6. **One comment, edited.** The comment is keyed on the job id (`<!-- neurarch-train:<job_id> -->`), so it is never confused with the plan comment and two runs on one pull request each keep their own. It is polled every 20 seconds up to `timeout-min`.
7. **A failed run never fails the check.** The trainer's error is reported as what it is: a fact about the model, not a verdict on the pull request.

### What the comment looks like

> ## Neurarch training run
>
> **`models/small_cnn.py:SmallCNN` on `mnist`** - :white_check_mark: finished
>
> | | |
> |---|---|
> | Job | `job_1788488972586_eoblzo` |
> | Config | 3 epochs, batch 64, lr 0.001, adamw |
> | Plan estimate | $0.0075 (roofline, before the run) |
> | Epochs | 3 |
> | Wall clock | 34s |
> | Cost | $0.0075 (measured from wall clock at this card's rate) |
>
> `val acc  ▁▆█`  98.65%
> `val loss █▃▁`  0.0477
>
> | epoch | train loss | train acc | val loss | val acc |
> |---:|---:|---:|---:|---:|
> | 1 | 0.1904 | 94.10% | 0.0921 | 97.12% |
> | 2 | 0.0651 | 97.99% | 0.0603 | 98.20% |
> | 3 | 0.0456 | 98.58% | 0.0477 | 98.65% |
>
> **val acc 98.65%** · val loss 0.0477
>
> Written to the outcome ledger: the next `/api/v1/plan` for this structure will quote this run.

The curve is a sparkline and a table rather than a picture on purpose. GitHub's comment sanitizer strips `src` from any `<img>` carrying a `data:` URI, in markdown and in raw HTML alike, and drops an inline `<svg>` entirely; this was measured against a real comment before the choice was made, not assumed.

### It costs money

Every accepted `/neurarch train` books a real GPU and bills the owner of the key in your repository secrets. A 3-epoch MNIST run is a few tenths of a cent; a long run on a big card is not. The write-access gate, the house-rules check and the endpoint's own refusal are all there because the cheapest run is the one that never started.

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
| `mode` | `pr` | `pr` plans the models a pull request touches and comments on it. `report` traces every listed model and upserts the weekly issue. `train` answers a `/neurarch train` comment; selected automatically on an `issue_comment` event. `init` writes the `.neurarch.yml` a repository does not have and opens a pull request adding it; it does nothing if one exists. |
| `config` | `.neurarch.yml` | Path to the config listing the models, relative to the repository root. |
| `api-key` | *(none)* | A Neurarch API key (`nrk_...`). Optional for the plan; keyless calls are rate-limited per hour and per IP. Also unlocks the server fix source, which is metered against the key owner's agent quota. **Required for `train`**, which spends the key owner's GPU budget. |
| `fix` | `false` | On a blocker, patch the model, re-trace, re-plan, and open a fix pull request if the new plan is legal and the parameter count is within 10 percent of the head. Needs `contents: write`. |
| `github-token` | `${{ github.token }}` | Token used to push the fix branch, open the fix pull request and add one line to the plan comment. A pull request opened with the default token gets no workflow run; pass a PAT to get one. |
| `timeout-min` | `20` | Train mode only. Minutes to keep polling the run and editing the comment. When it runs out the comment says the run is still going and how to poll it; the run is not cancelled and the job stays green. |
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
| `job-id` | In train mode, the id of the job submitted, or empty when nothing was. |
| `train-status` | In train mode: `none`, a `refused-...` reason, `timeout`, or the run's terminal status (`completed`, `error`, `cancelled`). |
| `init-status` | In init mode: `exists` (a config is already there, nothing was done), `nothing-found`, `opened`, `updated`, `no-token`, `push-failed`, `pr-failed` or `refused-same-branch`. |
| `init-pr-number` | In init mode, the number of the pull request adding the config. |
| `init-pr-url` | In init mode, the URL of that pull request. |
| `issue-number` | In report mode, the number of the weekly report issue created or updated. |
| `issue-url` | In report mode, the URL of that issue. |

## Privacy

The traced graph leaves your repository: layer structure and shapes only, never weights and never data. Nothing is published unless you ask.

`/neurarch train` sends the traced graph and the dataset name, and nothing else; the data is one of the three public datasets the endpoint offers and never comes from your repository.

Your **source files** are uploaded only by the server fix source, and only when you set both `fix: true` and an `api-key`: `POST /api/v1/fix` receives the model files `.neurarch.yml` names, because a patch cannot be written against a file nobody has read. It stores nothing and runs nothing. The exact patch source reads your files on the runner and sends nothing anywhere. See [neurarch.com/bot](https://neurarch.com/bot).

## License

MIT. See [LICENSE](./LICENSE).
