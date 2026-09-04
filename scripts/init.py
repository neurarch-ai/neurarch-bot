"""`mode: init`: write the `.neurarch.yml` a repository does not have, and open a pull request.

Everything before this mode assumed the config already existed, so a repository that installed
the Action got one log line telling it to write one by hand. The house rules in particular were
reachable only by somebody who already knew the `policy:` block existed, which means the feature
that turns the bot into *your* gate rather than ours was the one nobody switched on.

So: walk the repository for models, trace what can be traced, and write the config from what
came back. Two things this file will not do. It never invents a number: `max_params` is a round
number above the largest model that was actually measured here and `must_fit` is the smallest
card every measured model actually fits, and the pull request body says which lines are
measurement and which are suggestion. And it never writes to a branch anybody works on: the
config arrives as a pull request, because a config is a set of decisions and those are the
repository's to make, not ours.

Standard library only. The walk, the guesses, the derivation and both documents are pure
functions of what was found, so they are tested without torch, without a network and without
GitHub; `bot.py::run_init` is the part that traces, pushes and opens the pull request.
"""
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

TITLE = "Add a Neurarch config"
MARKER = "<!-- neurarch-bot-init -->"
BRANCH = "neurarch/add-config"
EXAMPLE_URL = "https://github.com/neurarch-ai/neurarch-bot/blob/main/examples/.neurarch.yml"
BOT_URL = "https://neurarch.com/bot"

# Directories a model is never found in, and walking them is how a first run turns into a
# five-minute one: a virtualenv alone carries thousands of `nn.Module` definitions that belong
# to torchvision, not to this repository.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".github", "node_modules", "__pycache__", ".venv", "venv", "env",
    "site-packages", "build", "dist", ".tox", ".nox", ".eggs", ".mypy_cache", ".pytest_cache",
    ".ipynb_checkpoints", ".idea", ".vscode", "tests", "test", "testing",
})
SKIP_FILES = re.compile(r"^(test_.*|.*_test|conftest|setup)\.py$")

MAX_FILES = 4000        # a walk, not an index
MAX_CANDIDATES = 40     # what one pull request body can honestly discuss
MAX_TRACE = 10          # each trace is a real import and a real forward pass

CLASS_RE = re.compile(r"^class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:", re.M)
MODULE_BASE = re.compile(r"\b(?:torch\.)?nn\.Module\b|\bLightningModule\b|\bpl\.LightningModule\b")

GPUS: Tuple[Tuple[str, str, str], ...] = (
    ("T4", "T4", "16GB"),
    ("A100-40GB", "A100_40", "40GB"),
    ("H100-80GB", "H100_80", "80GB"),
)

# The two most repositories want, and the reason is one bug wearing two hats: an activation in
# front of a loss that applies it again (Softmax before CrossEntropyLoss, Sigmoid before
# BCEWithLogitsLoss) trains, converges to something, and is wrong the whole time.
SUGGESTED_FORBID = ("softmax", "sigmoid")


# ---------------------------------------------------------------- discovery

def _readable(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def walk_files(root: str) -> Tuple[List[str], List[str]]:
    """Every candidate `.py` and every `*.neurarch.json` under `root`, as repository paths."""
    python: List[str] = []
    graphs: List[str] = []
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for name in sorted(filenames):
            seen += 1
            if seen > MAX_FILES:
                return python, graphs
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            if name.endswith(".neurarch.json"):
                graphs.append(rel)
            elif name.endswith(".py") and not SKIP_FILES.match(name):
                python.append(rel)
    return python, graphs


def class_blocks(source: str) -> List[Dict[str, Any]]:
    """Top-level classes that subclass `nn.Module`, each with its own body text.

    Only top-level ones (the regex is anchored at column zero): a module defined inside a
    function is not something `file.py:Name` can reach.
    """
    matches = [m for m in CLASS_RE.finditer(source) if MODULE_BASE.search(m.group(2))]
    all_starts = [m.start() for m in CLASS_RE.finditer(source)] + [len(source)]
    out = []
    for m in matches:
        end = next(s for s in all_starts if s > m.start())
        out.append({
            "name": m.group(1),
            "line": source.count("\n", 0, m.start()) + 1,
            "body": source[m.end():end],
        })
    return out


def _split_args(text: str) -> List[str]:
    args, depth, current = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current)
    return [a.strip() for a in args if a.strip()]


def required_arguments(body: str) -> List[str]:
    """Arguments `__init__` has no default for, other than `self`.

    A model is constructed with no arguments (`path: file.py:Class` is called as `Class()`), so
    a class whose constructor needs a `vocab_size` cannot be listed as it stands. It is still
    reported: the answer is a zero-argument factory next to it, and a config that silently
    dropped the model would never tell anybody that.
    """
    start = body.find("def __init__(")
    if start < 0:
        return []
    depth, end = 0, None
    for i in range(start + len("def __init__"), len(body)):
        if body[i] == "(":
            depth += 1
        elif body[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return []
    signature = body[start + len("def __init__("):end]
    required = []
    for arg in _split_args(signature):
        name = arg.split(":")[0].split("=")[0].strip()
        if not name or name in ("self", "cls") or name.startswith("*") or "=" in arg:
            continue
        required.append(name)
    return required


# Which layer decides the input shape, in the order it decides it. Not the order the layers
# appear in the file: a CNN whose first conv is built by a helper function still has an
# `nn.Linear` in its own body, and reading that one guesses a flat vector for a model that
# wants an image. Each kind is tried against the class body first, then the whole module.
_KINDS = ("embedding", "conv2d", "conv1d", "conv3d", "rnn", "linear")


def _first_arg(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern + r"\s*\(\s*(?:\w+\s*=\s*)?(\d+)\s*[,)]", text)
    return match.group(1) if match else None


def _specs_for(kind: str, text: str) -> Optional[List[str]]:
    if kind == "embedding":
        return ["1,32:long", "1,128:long"] if re.search(r"nn\.Embedding\s*\(", text) else None
    if kind == "conv2d":
        channels = _first_arg(r"Conv2d", text)
        if channels:
            return ["1,%s,32,32" % channels, "1,%s,28,28" % channels, "1,%s,224,224" % channels]
        # A conv built by a helper (`conv_bn_relu(3, 32)`) hides its channel count behind a
        # name, so the count is guessed from the two that cover almost every vision repository.
        return ["1,3,32,32", "1,1,28,28", "1,3,224,224"] if re.search(r"Conv2d\s*\(", text) else None
    if kind == "conv1d":
        channels = _first_arg(r"Conv1d", text)
        if channels:
            return ["1,%s,128" % channels]
        return ["1,1,128"] if re.search(r"Conv1d\s*\(", text) else None
    if kind == "conv3d":
        channels = _first_arg(r"Conv3d", text)
        if channels:
            return ["1,%s,8,32,32" % channels]
        return ["1,3,8,32,32"] if re.search(r"Conv3d\s*\(", text) else None
    if kind == "rnn":
        size = _first_arg(r"(?:LSTM|GRU|RNN)", text)
        return ["1,32,%s" % size] if size else None
    if kind == "linear":
        features = _first_arg(r"Linear", text)
        return ["1,%s" % features] if features else None
    return None


def guess_inputs(body: str, source: str = "") -> List[str]:
    """Input specs to try for one class, best first, read off the layers it constructs.

    A guess, and named as one everywhere it surfaces: the trace either accepts it or the model
    is written into the config commented out with the reason it did not. Nothing downstream
    ever treats a guessed shape as a measured one.
    """
    for kind in _KINDS:
        for text in (body, source):
            specs = _specs_for(kind, text) if text else None
            if specs:
                return specs
    return ["1,3,32,32"]


def building_block_of(name: str, blocks: List[Dict[str, Any]]) -> Optional[str]:
    """The class in the same file that constructs this one, if any.

    A transformer file holds `Block` and `TinyGPT`, and only one of them is a model: `Block` is
    a part, and tracing it with the model's input shape produces a confident plan for something
    nobody trains. If another class in the file constructs it, it is a part.
    """
    for other in blocks:
        if other["name"] == name:
            continue
        if re.search(r"\b%s\s*\(" % re.escape(name), other["body"]):
            return other["name"]
    return None


def discover(root: str) -> Dict[str, Any]:
    """Every model this repository might have, with a first guess at how to trace each."""
    python, graphs = walk_files(root)
    candidates: List[Dict[str, Any]] = []
    for rel in python:
        source = _readable(os.path.join(root, rel))
        if not source or ("nn.Module" not in source and "LightningModule" not in source):
            continue
        blocks = class_blocks(source)
        for block in blocks:
            if len(candidates) >= MAX_CANDIDATES:
                break
            candidates.append({
                "target": "%s:%s" % (rel, block["name"]),
                "file": rel,
                "name": block["name"],
                "inputs": guess_inputs(block["body"], source),
                "needs_args": required_arguments(block["body"]),
                "part_of": building_block_of(block["name"], blocks),
            })
    return {"candidates": candidates, "graphs": graphs, "python_files": len(python)}


# ---------------------------------------------------------------- derivation

def format_params(n: Optional[int]) -> str:
    if not isinstance(n, (int, float)) or n <= 0:
        return "?"
    if n >= 1e9:
        return "%.1fB" % (n / 1e9)
    if n >= 1e6:
        return "%.1fM" % (n / 1e6)
    if n >= 1e3:
        return "%.1fK" % (n / 1e3)
    return str(int(n))


def format_cap(n: int) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            value = n / size
            return "%d%s" % (int(value), unit) if value == int(value) else "%g%s" % (value, unit)
    return str(int(n))


def round_cap(largest: int, headroom: float = 1.2) -> str:
    """The smallest round number at least `headroom` above the biggest model measured.

    Round means one of 1, 2 or 5 times a power of ten, which is what somebody would have
    written by hand. The headroom is deliberately small: a cap ten times the largest model is
    a cap that never fires, and a rule that cannot fail is not a rule.
    """
    target = max(largest + 1, int(largest * headroom))
    scale = 1
    while scale <= 1e15:
        for mantissa in (1, 2, 5):
            value = mantissa * scale
            if value >= target:
                return format_cap(value)
        scale *= 10
    return format_cap(target)


def smallest_gpu(measured: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """The smallest card every measured model fits, as `(name, memory)`, or None if there isn't one."""
    if not measured:
        return None
    for name, key, memory in GPUS:
        fits = [(m.get("gpu_fits") or {}).get(key) for m in measured]
        if all(f is True for f in fits):
            return name, memory
    return None


def derive_policy(measured: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The suggested `policy:` block, and for each line where its number came from.

    `kind` is `measured` or `suggestion`, and it is carried all the way into the pull request
    body: a reader must never have to guess which of these numbers is a fact about their
    repository and which is our opinion about repositories in general.
    """
    out: Dict[str, Dict[str, Any]] = {}
    priced = [m for m in measured if isinstance(m.get("params"), int) and m["params"] > 0]
    if priced:
        biggest = max(priced, key=lambda m: m["params"])
        out["max_params"] = {
            "value": round_cap(biggest["params"]),
            "kind": "measured",
            "live": True,
            "because": "the largest model traced here is `%s` at %s params, so this blocks a "
                       "change that grows one past what this repository runs today."
                       % (biggest["target"], format_params(biggest["params"])),
        }
    gpu = smallest_gpu([m for m in measured if m.get("gpu_fits")])
    if gpu:
        out["must_fit"] = {
            "value": gpu[0],
            "kind": "measured",
            "live": True,
            "because": "every model traced here fits a %s (%s), the smallest card Neurarch "
                       "prices, so this blocks a change that stops fitting it."
                       % (gpu[0], gpu[1]),
        }
    out["forbid_types"] = {
        "value": "[%s]" % ", ".join(SUGGESTED_FORBID),
        "kind": "suggestion",
        "live": False,
        "because": "a Softmax in front of CrossEntropyLoss, or a Sigmoid in front of "
                   "BCEWithLogitsLoss, applies the activation twice: the model trains, "
                   "converges to something, and is wrong the whole time. Nothing was measured "
                   "for this line, so it arrives commented out.",
    }
    return out


# ---------------------------------------------------------------- the config

def _wrap(text: str, width: int = 96, prefix: str = "# ") -> List[str]:
    return textwrap.wrap(text, width=width, initial_indent=prefix, subsequent_indent=prefix,
                         break_long_words=False, break_on_hyphens=False) or [prefix.rstrip()]


def _quote(spec: str) -> str:
    """`1,32:long` has a colon in it, so YAML needs it quoted; `1,3,32,32` does not."""
    return '"%s"' % spec if ":" in spec else spec


def render_config(entries: List[Dict[str, Any]], derived: Dict[str, Dict[str, Any]],
                  version: str, graphs: Optional[List[str]] = None) -> str:
    """The `.neurarch.yml` itself: measured models, then the house rules derived from them."""
    listed = [e for e in entries if e.get("status") in ("measured", "traced")]
    skipped = [e for e in entries if e.get("status") == "skipped"]
    lines: List[str] = []
    lines += _wrap("Neurarch config, written by neurarch-bot %s. It lists the models found in "
                   "this repository and the house rules suggested for them." % version)
    lines += ["#"]
    lines += _wrap("The models are measurement: each one was constructed and traced at the "
                   "commit this was written from, and the numbers in the comments are what came "
                   "back. The policy block at the bottom is where a rule of yours goes; the "
                   "lines there now are a starting point, not a Neurarch default.")
    lines += ["#"]
    lines += _wrap("Every supported key, with a worked example: %s" % EXAMPLE_URL)
    lines += [""]
    lines += ["models:"]
    if not listed:
        lines += ["  # Nothing here traced. Fill in a `path:` and an `input:` and the bot takes it from there."]
    for entry in listed:
        lines += _wrap(_entry_note(entry), prefix="  # ")
        lines += ["  - path: %s" % entry["target"]]
        lines += ["    input: %s" % _quote(entry["input"])]
        lines += [""]
    if skipped:
        lines += _wrap("Found but not listed. Each of these is a model this bot could not "
                       "construct and trace on its own; the reason is on the line above it. "
                       "Fix the entry point or the input spec, uncomment, and it is planned "
                       "like the rest.", prefix="  # ")
        for entry in skipped:
            lines += _wrap(_entry_note(entry), prefix="  # ")
            lines += ["  # - path: %s" % entry["target"]]
            lines += ["  #   input: %s" % _quote(entry.get("input") or "1,3,32,32")]
        lines += [""]
    if graphs:
        lines += _wrap("This repository also carries %d traced graph%s (%s). A config lists "
                       "python entry points, so they are not listed here, but they were read "
                       "for the numbers below."
                       % (len(graphs), "" if len(graphs) == 1 else "s", ", ".join("`%s`" % g for g in graphs[:4])),
                       prefix="  # ")
        lines += [""]

    lines += _wrap("House rules. These are yours, not Neurarch's: the plan turns each line into "
                   "a blocker or a warning on every pull request, and quotes it back exactly as "
                   "written here. They need no API key, because a policy can only make a verdict "
                   "stricter and never more permissive.")
    lines += ["policy:"]
    if not any(rule.get("live") for rule in derived.values()):
        lines += ["  # Nothing was measured, so nothing here is set. Uncomment a line to set it."]
    for key in ("max_params", "must_fit", "max_layers", "forbid_types", "require_types",
                "max_memory_gb", "max_cost_usd"):
        rule = derived.get(key)
        if not rule:
            continue
        label = "Measured" if rule["kind"] == "measured" else "Suggestion, not measurement"
        lines += _wrap("%s: %s" % (label, rule["because"]), prefix="  # ")
        prefix = "  " if rule.get("live") else "  # "
        lines += ["%s%s: %s" % (prefix, key, rule["value"])]
    return "\n".join(lines).rstrip() + "\n"


def _entry_note(entry: Dict[str, Any]) -> str:
    if entry.get("status") == "measured":
        bits = ["%s layers" % entry["layers"]] if entry.get("layers") else []
        bits.append("%s params" % format_params(entry.get("params")))
        if entry.get("fits"):
            bits.append("fits a %s" % entry["fits"])
        return "%s: %s. Traced with input %s." % (entry["name"], ", ".join(bits), entry["input"])
    if entry.get("status") == "traced":
        return "%s: traced with input %s; the plan for it did not come back, so nothing about " \
               "its size fed the rules below." % (entry["name"], entry["input"])
    return "%s: %s" % (entry["name"], entry.get("reason") or "could not be traced.")


# ---------------------------------------------------------------- the pull request

def pr_body(entries: List[Dict[str, Any]], derived: Dict[str, Dict[str, Any]], version: str,
            config_name: str = ".neurarch.yml", graphs: Optional[List[str]] = None,
            run_url: Optional[str] = None, api_host: str = "www.neurarch.com",
            searched: Optional[int] = None) -> str:
    """The body that explains the file, line by line, and says what was measured.

    Every claim in here is one the reader can check against the run log this links to, which
    is the point: a config generated by a tool is worth exactly as much as the reader's
    ability to disagree with it line by line.
    """
    measured = [e for e in entries if e.get("status") == "measured"]
    traced = [e for e in entries if e.get("status") == "traced"]
    skipped = [e for e in entries if e.get("status") == "skipped"]
    parts = [MARKER, "## %s" % TITLE, ""]
    parts.append("This repository has no `%s`, so neurarch-bot had nothing to plan when a pull "
                 "request touched a model. This adds one." % config_name)
    parts.append("")
    parts.append("It was written by walking the repository%s, constructing every `nn.Module` "
                 "found and tracing it. Nothing here is a claim about what you meant: the "
                 "models are measurement, the rules are derived from that measurement or "
                 "marked as a suggestion, and each is labelled below."
                 % (" (%d python files, outside tests, build directories and virtualenvs)" % searched
                    if searched else ""))
    parts.append("")

    parts.append("### The models, and what they measured")
    parts.append("")
    if measured or traced:
        parts.append("| Model | Input | Layers | Params | GPU fit |")
        parts.append("|---|---|---|---|---|")
        for entry in measured + traced:
            parts.append("| `%s` | `%s` | %s | %s | %s |" % (
                entry["target"], entry["input"], entry.get("layers") or "?",
                format_params(entry.get("params")), entry.get("fits") or "not planned"))
    else:
        parts.append("Nothing traced. The models below are listed commented out, with the "
                     "reason on each, and the file is worth merging only once one of them works.")
    parts.append("")
    parts.append("The input spec is a guess read off the first layer each model constructs "
                 "(an `nn.Embedding` means token ids, a `Conv2d(3, ...)` means three channels), "
                 "and it is a guess that the trace then either accepted or did not. Change it "
                 "and the plan changes with it.")
    parts.append("")

    if skipped:
        parts.append("### Found, not listed")
        parts.append("")
        for entry in skipped:
            parts.append("- `%s`: %s" % (entry["target"], entry.get("reason") or "could not be traced."))
        parts.append("")
        parts.append("They are in the file, commented out, so nothing found is silently dropped.")
        parts.append("")

    if graphs:
        parts.append("### Traced graphs already in the repository")
        parts.append("")
        parts.append("%s. A config lists python entry points, so these are not entries, but "
                     "they were read for the numbers below."
                     % ", ".join("`%s`" % g for g in graphs[:8]))
        parts.append("")

    parts.append("### The house rules, line by line")
    parts.append("")
    parts.append("This is the part worth arguing with. A policy line is a decision your team "
                 "makes; the plan enforces it on every pull request and quotes it back as you "
                 "wrote it. The bot suggests these three and nothing else.")
    parts.append("")
    for key in ("max_params", "must_fit", "max_layers", "forbid_types", "require_types",
                "max_memory_gb", "max_cost_usd"):
        rule = derived.get(key)
        if not rule:
            continue
        state = "on" if rule.get("live") else "commented out"
        label = "**measured**" if rule["kind"] == "measured" else "**a suggestion, not a measurement**"
        parts.append("- `%s: %s` (%s, %s). %s" % (key, rule["value"], state, label, rule["because"]))
    parts.append("")
    parts.append("Every other supported key, with what it blocks and what it warns on, is in "
                 "[the annotated example](%s)." % EXAMPLE_URL)
    parts.append("")

    parts.append("### What happens after this merges")
    parts.append("")
    parts.append("- Every pull request that touches one of these files gets a plan comment: "
                 "params, cost, GPU fit, whether it will run, and the deltas against the base branch.")
    parts.append("- A blocker turns the check red. That includes your own policy lines, and the "
                 "comment names the line of `%s` each one came from." % config_name)
    parts.append("- Nothing in this pull request spends anything, and nothing was sent anywhere "
                 "except the traced structure (layers and shapes, never weights or data) to %s "
                 "to be planned." % api_host)
    parts.append("")
    footer = "<sub>neurarch-bot v%s. [How it works](%s)" % (version, BOT_URL)
    if run_url:
        footer += " · [run log](%s)" % run_url
    footer += "</sub>"
    parts.append(footer)
    return "\n".join(parts) + "\n"


def branch_name(base: str, default_branch: Optional[str]) -> str:
    """`neurarch/add-config`, or one branch per base when the run is not on the default branch.

    Two runs on two branches must not fight over one branch and one pull request: the second
    would force-push the first's commit away and then fail to change the open pull request's
    base.
    """
    if not base or base == (default_branch or base):
        return BRANCH
    return "%s-%s" % (BRANCH, re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-").lower())
