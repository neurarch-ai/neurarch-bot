"""The job half of the plan comment: what the training job reserves, and whether it still fits.

A pull request that changes a model is a change to a GPU reservation somebody already made.
The person who owns that bill reads one thing first: at the batch and sequence length this job
actually trains at, does the peak still fit the card it reserves, and if not, what does. That
is what this module answers, before the structural checks.

The job is read from `.neurarch.yml`, per model or as a top-level default:

    job:
      gpu: A10G                                  # the card the training job reserves
      batch: configs/pretrain.yaml:batch_size    # a number, or `file:key` read at each commit
      seq_len: configs/pretrain.yaml:seq_len
      grad_accum: configs/pretrain.yaml:grad_accum

A `file:key` reference is read at the commit being planned, on both sides of the pull request,
so a change to the training config alone moves the number and the base is sized with the batch
it actually had. The arithmetic is not here: `size_job.mjs` runs the vendored plan bundle.

Standard library only (plus pyyaml when present, as the config reader already is).
"""
import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE_SCRIPT = os.path.join(HERE, "size_job.mjs")
VENDOR_VERSION = os.path.join(HERE, "vendor", "VERSION")
JOB_KEYS = ("gpu", "batch", "seq_len", "grad_accum", "attention", "epochs", "samples")
NUMERIC_KEYS = ("batch", "seq_len", "grad_accum", "epochs", "samples")
_REF = re.compile(r"^(?P<file>[^:\s]+\.(?:ya?ml|json)):(?P<key>[A-Za-z0-9_.\-]+)$")


# ---------------------------------------------------------------- config

def parse_job(config: Dict[str, Any], model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The job for one model: the top-level `job:` block, then the model's own on top, per key."""
    merged: Dict[str, Any] = {}
    top = config.get("job")
    if isinstance(top, dict):
        merged.update({k: v for k, v in top.items() if k in JOB_KEYS})
    own = model.get("job")
    if isinstance(own, dict):
        merged.update({k: v for k, v in own.items() if k in JOB_KEYS})
    return merged or None


def job_files(job: Optional[Dict[str, Any]]) -> List[str]:
    """Files the job reads its numbers from: a change to one of them is a change to the job."""
    out: List[str] = []
    for v in (job or {}).values():
        m = _REF.match(str(v).strip()) if isinstance(v, str) else None
        if m:
            out.append(os.path.normpath(m.group("file")))
    return sorted(set(out))


def _load_structured(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def _dig(data: Any, key: str) -> Any:
    for part in key.split("."):
        if not isinstance(data, dict) or part not in data:
            raise KeyError(key)
        data = data[part]
    return data


def resolve_job(job: Optional[Dict[str, Any]], root: str) -> Tuple[Dict[str, Any], List[str]]:
    """Every `file:key` reference read at `root`; numbers coerced. Returns (job, problems)."""
    out: Dict[str, Any] = {}
    problems: List[str] = []
    for k, v in (job or {}).items():
        sources = ""
        if isinstance(v, str) and _REF.match(v.strip()):
            m = _REF.match(v.strip())
            rel, key = m.group("file"), m.group("key")
            try:
                v = _dig(_load_structured(os.path.join(root, rel)), key)
                sources = "%s:%s" % (rel, key)
            except FileNotFoundError:
                problems.append("`job.%s` reads `%s`, which does not exist at this commit" % (k, rel))
                continue
            except KeyError:
                problems.append("`job.%s` reads `%s:%s`, and that key is not in the file" % (k, rel, key))
                continue
            except (ValueError, OSError) as e:
                problems.append("`job.%s` could not read `%s`: %s" % (k, rel, e))
                continue
            out.setdefault("_sources", {})[k] = sources
        if k in NUMERIC_KEYS:
            try:
                n = float(v)
            except (TypeError, ValueError):
                problems.append("`job.%s` is `%s`, not a number" % (k, v))
                continue
            if n <= 0 or n != int(n):
                problems.append("`job.%s` is `%s`, not a positive whole number" % (k, v))
                continue
            out[k] = int(n)
        else:
            out[k] = str(v)
    return out, problems


# ---------------------------------------------------------------- sizing

def size_job(head_graph: str, base_graph: Optional[str], job: Dict[str, Any],
             base_job: Optional[Dict[str, Any]], history: Optional[List[Dict[str, Any]]],
             tmp: str) -> Dict[str, Any]:
    """Run `size_job.mjs`. Never raises: a sizing that could not run says why."""
    node = shutil.which("node")
    if not node:
        return {"ok": False, "error": "node is not on the runner, so the job could not be sized"}
    req = {"head": head_graph, "base": base_graph,
           "job": {k: v for k, v in job.items() if not k.startswith("_")},
           "base_job": ({k: v for k, v in base_job.items() if not k.startswith("_")}
                        if base_job is not None else None),
           "history": history or []}
    req_path = os.path.join(tmp, "neurarch-size-request.json")
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(req, f)
    try:
        r = subprocess.run([node, SIZE_SCRIPT, req_path], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "sizing the job timed out"}
    try:
        data = json.loads(r.stdout)
    except ValueError:
        return {"ok": False, "error": (r.stderr or r.stdout or "no output").strip().splitlines()[-1:][0]
                if (r.stderr or r.stdout) else "no output"}
    return data if isinstance(data, dict) else {"ok": False, "error": "sizing returned no object"}


def bundle_version() -> Optional[str]:
    try:
        with open(VENDOR_VERSION, "r", encoding="utf-8") as f:
            line = f.read().strip().splitlines()[0]
        return line or None
    except (OSError, IndexError):
        return None


# ---------------------------------------------------------------- verdict

def should_fail(entries: List[Dict[str, Any]], fail_on_warn: bool, fail_on_no_fit: bool) -> bool:
    """Red when a model will not run, when its job no longer fits the card it reserves (unless
    switched off), or on a warning when asked to be that strict."""
    if not all(e.get("legal") for e in entries):
        return True
    if fail_on_warn and sum(int(e.get("warnings") or 0) for e in entries) > 0:
        return True
    return fail_on_no_fit and any(fits(e.get("sizing")) is False for e in entries)


def fits(sizing: Optional[Dict[str, Any]]) -> Optional[bool]:
    """True / False against the reserved card; None when there is no card or no sizing."""
    if not isinstance(sizing, dict) or not sizing.get("ok") or not sizing.get("card"):
        return None
    head = sizing.get("head") or {}
    return head.get("fits") if isinstance(head.get("fits"), bool) else None


def no_longer_fits(sizing: Optional[Dict[str, Any]]) -> bool:
    base = (sizing or {}).get("base") or {}
    return fits(sizing) is False and base.get("fits") is True


def memory_moved(sizing: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(sizing, dict) or not sizing.get("ok"):
        return False
    head, base = sizing.get("head") or {}, sizing.get("base")
    if not isinstance(base, dict):
        return False
    return abs(float(head.get("peakGb") or 0) - float(base.get("peakGb") or 0)) >= 0.05 \
        or head.get("fits") != base.get("fits")


# ---------------------------------------------------------------- formatting

def card_label(name: Optional[str]) -> str:
    """"A10G (24GB)" -> "A10G 24 GB"."""
    if not name:
        return "?"
    m = re.match(r"^(.*?)\s*\((\d+)GB\)$", name)
    return "%s %s GB" % (m.group(1), m.group(2)) if m else name


def gb(x: Optional[float]) -> str:
    return "?" if x is None else "%.1f GB" % float(x)


def signed_gb(x: float) -> str:
    return "%s%.1f GB" % ("+" if x >= 0 else "-", abs(x))


def money(usd: Optional[float]) -> str:
    if usd is None:
        return "?"
    if usd < 0.005:
        return "<$0.01"
    if usd < 100:
        return "$%.2f" % usd
    return "$%s" % format(int(round(usd)), ",")


def duration(sec: Optional[float]) -> str:
    if sec is None:
        return "?"
    if sec < 90:
        return "%ds" % int(round(sec))
    if sec < 5400:
        return "%d min" % int(round(sec / 60))
    return "%.1f h" % (sec / 3600)


def at_batch(side: Dict[str, Any], accum: Optional[int] = None) -> str:
    s = "batch %d" % side.get("batch", 0)
    if side.get("seqLen"):
        s += " x %s tokens" % format(int(side["seqLen"]), ",")
    if accum and accum > 1:
        s += ", %d accumulation steps for an effective batch of %d" % (accum, side.get("batch", 0) * accum)
    return s


def accumulation_split(target: int, max_batch: int) -> Tuple[int, int]:
    """(micro, steps) with micro <= max_batch and micro x steps == target when a divisor allows.

    The divisor is preferred so the effective batch the job was tuned at does not move; when the
    only divisors are far below the cap (a prime target), the cap itself and the rounded-up step
    count are returned and the caller prints the effective batch that gives.
    """
    if max_batch >= target:
        return target, 1
    best = max(d for d in range(1, max_batch + 1) if target % d == 0)
    if best * 2 >= max_batch:
        return best, target // best
    steps = -(-target // max_batch)
    return max_batch, steps


# ---------------------------------------------------------------- render

def badge(sizing: Optional[Dict[str, Any]]) -> Optional[str]:
    f = fits(sizing)
    if f is None:
        return None
    card = card_label((sizing.get("card") or {}).get("name"))
    if f:
        return ":white_check_mark: fits %s" % card
    if no_longer_fits(sizing):
        return ":no_entry: no longer fits %s" % card
    return ":no_entry: does not fit %s" % card


def render_lead(sizing: Optional[Dict[str, Any]], job: Optional[Dict[str, Any]],
                problems: Optional[List[str]] = None, config_name: str = ".neurarch.yml") -> List[str]:
    """The memory block that opens a model's section. Empty when there is nothing to say."""
    out: List[str] = []
    if not isinstance(sizing, dict):
        return out
    if not sizing.get("ok"):
        out.append("_Training memory could not be sized on the runner (%s); the structural checks "
                   "below still ran._" % (sizing.get("error") or "unknown error"))
        return out
    job = job or {}
    head = sizing.get("head") or {}
    base = sizing.get("base")
    card = sizing.get("card")
    accum = job.get("grad_accum")
    base_accum = (sizing.get("baseJob") or {}).get("grad_accum")
    headroom = sizing.get("headroom") or 1.15

    peak = "**Peak training memory: %s** at %s" % (gb(head.get("peakGb")), at_batch(head, accum))
    if head.get("activationGb") is not None:
        peak += " (weights and optimizer state %s, activations %s)." % (
            gb(head.get("footprintGb")), gb(head.get("activationGb")))
    else:
        peak += ". Activations are not in it: no shape propagated through this graph."
    out.append(peak)
    out.append("")

    bullets: List[str] = []
    label = card_label(card.get("name")) if card else None
    if card:
        if head.get("fits"):
            bullets.append("**Reserved card: %s. Fits**: the job needs %s of its %d GB with the %.2fx "
                           "allocator headroom." % (label, gb(head.get("needGb")), card["gb"], headroom))
        else:
            bullets.append("**Reserved card: %s. Does not fit**: the job needs %s with the %.2fx "
                           "allocator headroom, and the card has %d GB." % (label, gb(head.get("needGb")),
                                                                          headroom, card["gb"]))
    elif sizing.get("cardError"):
        bullets.append("**Reserved card: not read.** %s." % sizing["cardError"])
    else:
        bullets.append("**Reserved card: not set.** Add `job: {gpu: A10G, batch: 32}` to this model in "
                       "`%s` and the check turns red when a change stops fitting it." % config_name)

    if isinstance(base, dict):
        delta = float(head.get("peakGb") or 0) - float(base.get("peakGb") or 0)
        moved = signed_gb(delta) if abs(delta) >= 0.05 else "same peak"
        line = "**vs base:** %s (base %s at %s)" % (moved, gb(base.get("peakGb")),
                                                  at_batch(base, base_accum))
        if card and base.get("fits") is True and head.get("fits") is False:
            line += ", **no longer fits %s**" % label
        elif card and base.get("fits") is False and head.get("fits") is True:
            line += ", **now fits %s**" % label
        elif card and base.get("fits") is False and head.get("fits") is False:
            line += ", and the base did not fit %s either" % label
        bullets.append(line + ".")

    if card:
        mb = head.get("maxBatch")
        tokens = " at %s tokens" % format(int(head["seqLen"]), ",") if head.get("seqLen") else ""
        if mb is None:
            if head.get("activationGb") is None:
                bullets.append("**Largest batch that fits %s:** unknown, because activations are not "
                               "sized for this graph." % label)
            else:
                bullets.append("**Largest batch that fits %s:** none; the weights and optimizer state "
                               "alone do not fit it." % label)
        else:
            line = "**Largest batch that fits %s:** %s%s." % (label, format(mb, ","), tokens)
            target = int(head.get("batch") or 0) * int(accum or 1)
            if not head.get("fits") and target > 0:
                micro, steps = accumulation_split(target, mb)
                if steps > 1:
                    eff = micro * steps
                    line += " Batch %d with %d gradient-accumulation steps keeps the effective batch at %d%s." % (
                        micro, steps, eff, "" if eff == target else " (the job had %d)" % target)
            bullets.append(line)

    smallest = head.get("smallestFitGpu")
    if smallest:
        line = "**Smallest card that holds it:** %s" % card_label(smallest)
        if card and head.get("fits") and head.get("smallestFitGb") is not None \
                and head["smallestFitGb"] < card["gb"]:
            line += ", smaller than the %d GB this job reserves" % card["gb"]
        bullets.append(line + ".")
    elif head.get("twoCardSplit"):
        bullets.append("**Smallest card that holds it:** none of the listed cards alone; two %s can, "
                       "split across them." % card_label(head["twoCardSplit"].get("gpu")))
    else:
        bullets.append("**Smallest card that holds it:** none of the listed cards, 80 GB included.")

    if head.get("costUsd") is not None:
        cost = "**Cost:** about %s and %s on %s for %d epochs over %s samples%s" % (
            money(head["costUsd"]), duration(head.get("trainSec")), card_label(head.get("costCard")),
            head.get("epochs") or 10, format(int(head.get("samples") or 50000), ","),
            " (assumed; set `job.samples`)" if head.get("samplesAssumed") else "")
        if card and not head.get("fits") and head.get("costCard") != card.get("name"):
            cost += ", priced on the card that holds it"
        bullets.append(cost + ".")

    if sizing.get("historyLine"):
        bullets.append(sizing["historyLine"] + ".")
    for p in problems or []:
        bullets.append(":warning: %s." % p)
    out.extend("- %s" % b for b in bullets)
    return out
