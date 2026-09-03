# neurarch-bot

[![CI](https://github.com/neurarch-ai/neurarch-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/neurarch-ai/neurarch-bot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Try Neurarch](https://img.shields.io/badge/Neurarch-try_it-7c3aed)](https://neurarch.com/bot)

**Dependabot for models.** On every pull request that touches a model, neurarch-bot traces the model on your branch and on the base branch, and posts one comment saying what changed: params, cost, GPU fit, and whether the thing will run at all. A blocker turns the check red, so a model that cannot forward-pass never reaches a GPU.

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

Per model you may also set `extra_inputs` (a list of further `--input` specs, for a forward that takes more than one tensor), `depth` (stop descending at this module depth) and `name`.

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

## Inputs

| Input | Default | What it does |
|---|---|---|
| `config` | `.neurarch.yml` | Path to the config listing the models, relative to the repository root. |
| `api-key` | *(none)* | A Neurarch API key (`nrk_...`). Optional; keyless calls are rate-limited per hour and per IP. |
| `fail-on-warn` | `false` | Fail the check on warnings, not just blockers. |
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

## Privacy

The traced graph leaves your repository: layer structure and shapes only, never weights and never data. Your source files are not uploaded. Nothing is published unless you ask. See [neurarch.com/bot](https://neurarch.com/bot).

## License

MIT. See [LICENSE](./LICENSE).
