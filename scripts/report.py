"""The weekly-report half of neurarch-bot: turn planned models into one issue body.

Report mode traces every model in the config at the current commit, with no base and no
diff against a branch. What it compares against instead is the *previous report*, which is
why the body carries a machine-readable block on its last line: the numbers this run
produced, so the next run can read them back out of the issue it is about to overwrite.

Standard library only, no torch, no network: everything here is a pure function over the
plan responses `plan_request` already fetched, so the body, the round trip of the embedded
block, the drift check and the week-over-week diff are all testable with no model.
"""
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

MARKER = "<!-- neurarch-report -->"
DATA_OPEN = "<!-- neurarch-report-data "
DATA_CLOSE = " -->"
TITLE = "Neurarch weekly model report"
DATA_VERSION = 1
DRIFT_TOLERANCE = 0.01
BOT_URL = "https://neurarch.com/bot"

# Documents that may carry a parameter count for a model in the same directory, in the order
# a more specific name beats a more general one.
DOC_NAMES = ("MODEL_CARD.md", "model_card.md", "ModelCard.md", "README.md", "readme.md")

_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*([KMBkmb])?"
_PARAM_WORD = r"param(?:eter)?s?(?:\s+count)?"
_PARAM_PATTERNS = (
    # "Parameters: 1,200,000" / "| Params | 1.2M |" / "parameter count = 1200000"
    re.compile(_PARAM_WORD + r"\s*[:=|]\s*~?\s*" + _NUM, re.IGNORECASE),
    # "1.2M parameters" / "~971.9K params"
    re.compile(r"~?\s*" + _NUM + r"\s+(?:total\s+|trainable\s+)?" + _PARAM_WORD, re.IGNORECASE),
)
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}


# ---------------------------------------------------------------- formatting

def fmt_params(n: Optional[float]) -> str:
    if n is None:
        return "?"
    n = float(n)
    if n >= 1e9:
        return "%.1fB" % (n / 1e9)
    if n >= 1e6:
        return "%.1fM" % (n / 1e6)
    if n >= 1e3:
        return "%.1fK" % (n / 1e3)
    return "%d" % int(round(n))


def fmt_money(usd: Optional[float]) -> str:
    if usd is None:
        return "?"
    if usd < 0.005:
        return "<$0.01"
    if usd < 10:
        return "$%.2f" % usd
    return "$%d" % int(round(usd))


def fmt_shape(shape: Any) -> str:
    if not isinstance(shape, list) or not shape:
        return "?"
    return "[%s]" % ",".join(str(x) for x in shape)


def fmt_pct(delta: float, base: float) -> str:
    if not base:
        return ""
    return " (%+.0f%%)" % (100.0 * delta / base)


# ---------------------------------------------------------------- one row per model

def model_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The compact record for one model: what the table prints and what the block carries."""
    row: Dict[str, Any] = {
        "target": entry.get("target"),
        "input": entry.get("input"),
        "status": entry.get("status"),
        "legal": bool(entry.get("legal")),
        "blockers": int(entry.get("blockers") or 0),
        "warnings": int(entry.get("warnings") or 0),
    }
    plan = entry.get("plan")
    if isinstance(plan, dict):
        model = plan.get("model") or {}
        cost = plan.get("cost") or {}
        row["name"] = model.get("name")
        row["layers"] = model.get("layers")
        row["params"] = model.get("params")
        row["fingerprint"] = model.get("fingerprint")
        row["inputShape"] = model.get("inputShape")
        row["outputShape"] = model.get("outputShape")
        row["fitsGpu"] = cost.get("fitsGpu")
        est = cost.get("estCostUsd")
        row["estCostUsd"] = round(float(est), 4) if isinstance(est, (int, float)) else None
    if entry.get("error"):
        row["error"] = (entry["error"].splitlines() or [""])[-1]
    return row


def render_table(rows: List[Dict[str, Any]]) -> str:
    """Every model, one line each. A model that did not trace keeps its line and says so."""
    out = ["| Model | Layers | Params | Shape | Will it run | GPU fit | Est. cost | Fingerprint |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        target = "`%s`" % r.get("target")
        if r.get("status") != "planned":
            reason = "could not trace" if r.get("status") == "trace_failed" else "no plan"
            out.append("| %s | - | - | - | :no_entry: %s | - | - | - |" % (target, reason))
            continue
        blockers, warnings = r.get("blockers") or 0, r.get("warnings") or 0
        if blockers:
            verdict = ":no_entry: no (%d blocker%s)" % (blockers, "" if blockers == 1 else "s")
        elif warnings:
            verdict = ":warning: yes (%d warning%s)" % (warnings, "" if warnings == 1 else "s")
        else:
            verdict = ":white_check_mark: yes"
        shape = "`%s` -> `%s`" % (fmt_shape(r.get("inputShape")), fmt_shape(r.get("outputShape")))
        out.append("| %s | %s | %s | %s | %s | %s | ~%s | `%s` |" % (
            target, r.get("layers") if r.get("layers") is not None else "?", fmt_params(r.get("params")),
            shape, verdict, r.get("fitsGpu") or "none of the priced GPUs",
            fmt_money(r.get("estCostUsd")), r.get("fingerprint") or "?"))
    return "\n".join(out)


# ---------------------------------------------------------------- findings

def _findings(entry: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    run = ((entry.get("plan") or {}).get("run") or {})
    items = run.get(key)
    return [f for f in items if isinstance(f, dict)] if isinstance(items, list) else []


def render_findings(entries: List[Dict[str, Any]]) -> str:
    """Every blocker and warning across every model, quoted in full, grouped by model."""
    blockers = sum(len(_findings(e, "blockers")) for e in entries)
    warnings = sum(len(_findings(e, "warnings")) for e in entries)
    untraced = [e for e in entries if e.get("status") != "planned"]
    out = ["### Blockers and warnings", ""]
    if not blockers and not warnings and not untraced:
        out.append("None. Every model in the config traces, forward-passes, and comes back clean.")
        return "\n".join(out)
    out.append("%d blocker%s and %d warning%s across %d model%s." % (
        blockers, "" if blockers == 1 else "s", warnings, "" if warnings == 1 else "s",
        len(entries), "" if len(entries) == 1 else "s"))
    out.append("")
    for entry in entries:
        rows = [(":no_entry: blocker:", f) for f in _findings(entry, "blockers")]
        rows += [(":warning: warning:", f) for f in _findings(entry, "warnings")]
        if entry.get("status") != "planned":
            out.append("#### `%s`" % entry.get("target"))
            out.append("")
            out.append("- :no_entry: **%s** - %s" % (
                "could not trace" if entry.get("status") == "trace_failed" else "no plan came back",
                (entry.get("error") or "unknown error").splitlines()[-1]))
            out.append("")
            continue
        if not rows:
            continue
        out.append("#### `%s`" % entry.get("target"))
        out.append("")
        for label, f in rows:
            line = "- %s **%s**" % (label, f.get("title") or "finding")
            detail = f.get("detail")
            if detail:
                line += " - %s" % detail
            out.append(line)
            if f.get("fix"):
                out.append("  - Fix: %s" % f["fix"])
        out.append("")
    return "\n".join(out).rstrip()


# ---------------------------------------------------------------- the embedded block

def _escape(text: str) -> str:
    """No `-->` can survive inside an HTML comment; escaping `>` keeps the JSON valid."""
    return text.replace(">", "\\u003e")


def embed_data(rows: List[Dict[str, Any]], commit: Optional[str] = None,
               generated: Optional[str] = None) -> str:
    payload = {"version": DATA_VERSION, "generated": generated or _now_iso(),
               "commit": commit, "models": rows}
    return DATA_OPEN + _escape(json.dumps(payload, separators=(",", ":"), sort_keys=True)) + DATA_CLOSE


def parse_report_data(body: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read the block back out of a previous issue body. None when there is nothing to read."""
    if not body:
        return None
    m = re.search(re.escape(DATA_OPEN) + r"(.*?)" + re.escape(DATA_CLOSE), body, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        return None
    return data


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pretty_date(iso: Optional[str]) -> str:
    if not iso:
        return "the last report"
    return iso.split("T")[0]


# ---------------------------------------------------------------- week over week

def diff_reports(previous: Optional[Dict[str, Any]], rows: List[Dict[str, Any]]) -> List[str]:
    """One bullet per model that is new, gone, or numerically different from last week."""
    if not previous:
        return []
    old = {r.get("target"): r for r in previous.get("models") or [] if isinstance(r, dict)}
    new = {r.get("target"): r for r in rows}
    lines: List[str] = []
    for target, r in new.items():
        if target not in old:
            lines.append("`%s`: **new in this report** (%s params)" % (target, fmt_params(r.get("params"))))
            continue
        o = old[target]
        parts: List[str] = []
        op, np_ = o.get("params"), r.get("params")
        if isinstance(op, (int, float)) and isinstance(np_, (int, float)) and op != np_:
            parts.append("params %s -> %s%s" % (fmt_params(op), fmt_params(np_), fmt_pct(np_ - op, op)))
        ol, nl = o.get("layers"), r.get("layers")
        if isinstance(ol, int) and isinstance(nl, int) and ol != nl:
            parts.append("layers %d -> %d" % (ol, nl))
        if o.get("legal") and not r.get("legal"):
            parts.append("**will no longer run** (%d blocker(s))" % (r.get("blockers") or 0))
        elif r.get("legal") and not o.get("legal"):
            parts.append("**runs again** (was blocked)")
        ow, nw = o.get("warnings") or 0, r.get("warnings") or 0
        if ow != nw:
            parts.append("warnings %d -> %d" % (ow, nw))
        oc, nc = o.get("estCostUsd"), r.get("estCostUsd")
        if isinstance(oc, (int, float)) and isinstance(nc, (int, float)) and abs(nc - oc) >= 0.005:
            parts.append("est. cost %s -> %s" % (fmt_money(oc), fmt_money(nc)))
        of, nf = o.get("fingerprint"), r.get("fingerprint")
        if of and nf and of != nf and not parts:
            # The structure moved but every number this report prints stayed put.
            parts.append("fingerprint `%s` -> `%s`, same params and shape" % (of, nf))
        elif of and nf and of != nf:
            parts.append("fingerprint `%s` -> `%s`" % (of, nf))
        if parts:
            lines.append("`%s`: %s" % (target, "; ".join(parts)))
    for target in old:
        if target not in new:
            lines.append("`%s`: **no longer listed** in the config" % target)
    return lines


def render_changes(previous: Optional[Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
    out = ["### Changed since the last report", ""]
    if not previous:
        out.append("This is the first report, so there is nothing to compare against. "
                   "Next week's report diffs against the numbers below.")
        return "\n".join(out)
    lines = diff_reports(previous, rows)
    stamp = _pretty_date(previous.get("generated"))
    if not lines:
        out.append("Nothing changed since %s: same models, same params, same verdicts." % stamp)
        return "\n".join(out)
    out.append("Against the report of %s:" % stamp)
    out.append("")
    out += ["- %s" % line for line in lines]
    return "\n".join(out)


# ---------------------------------------------------------------- documentation drift

def parse_param_claims(text: str) -> List[Tuple[float, str]]:
    """Every parameter count written in prose or a table cell, with the phrase it came from."""
    found: List[Tuple[int, float, str]] = []
    seen = set()
    for pattern in _PARAM_PATTERNS:
        for m in pattern.finditer(text):
            raw, suffix = m.group(1), m.group(2)
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue
            if suffix:
                value *= _SUFFIX[suffix.lower()]
            elif "." in raw:
                # "1.2 params" with no suffix is not a count of anything.
                continue
            if value <= 0:
                continue
            key = (round(value), m.start())
            if key in seen:
                continue
            seen.add(key)
            found.append((m.start(), value, " ".join(m.group(0).split())))
    found.sort(key=lambda c: c[0])
    return [(value, phrase) for _, value, phrase in found]


def _needles(model: Dict[str, Any]) -> List[str]:
    target = model.get("target") or ""
    file = model.get("file") or ""
    base = os.path.basename(file)
    out = [base, os.path.splitext(base)[0]]
    if ":" in target:
        out.append(target.rsplit(":", 1)[1])
    return [n for n in out if n]


def doc_drift(model: Dict[str, Any], traced_params: Optional[float], root: str,
              other_models: Optional[List[Dict[str, Any]]] = None,
              tolerance: float = DRIFT_TOLERANCE) -> Optional[Dict[str, Any]]:
    """A parameter count written next to the model that disagrees with the traced one.

    Scoping rule, so one card shared by a folder does not accuse every model in it: a doc that
    names other listed models and not this one is skipped. A doc that names this one (or names
    no model at all) is read whole, and if *any* number in it agrees within tolerance the doc is
    treated as right and nothing is reported. Otherwise the closest number is the one quoted,
    because that is the one someone meant to keep up to date.
    """
    if not isinstance(traced_params, (int, float)) or traced_params <= 0:
        return None
    directory = os.path.dirname(model.get("file") or "")
    mine = _needles(model)
    theirs = [n for m in (other_models or []) for n in _needles(m) if n not in mine]
    for name in DOC_NAMES:
        path = os.path.join(root, directory, name) if directory else os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        names_me = any(n in text for n in mine)
        if not names_me and any(n in text for n in theirs):
            continue
        claims = parse_param_claims(text)
        if not claims:
            continue
        if any(abs(v - traced_params) / traced_params <= tolerance for v, _ in claims):
            return None
        value, phrase = min(claims, key=lambda c: abs(c[0] - traced_params))
        return {
            "target": model.get("target"),
            "doc": os.path.join(directory, name) if directory else name,
            "claimed": value,
            "traced": float(traced_params),
            "offBy": 100.0 * (value - traced_params) / traced_params,
            "phrase": phrase,
        }
    return None


def render_drift(drift: List[Dict[str, Any]]) -> str:
    out = ["### Documentation drift", ""]
    if not drift:
        out.append("No parameter count written next to a model disagrees with the traced one.")
        return "\n".join(out)
    out.append("A parameter count in the docs disagrees with what the trace measured "
               "by more than %d%%:" % int(round(DRIFT_TOLERANCE * 100)))
    out.append("")
    for d in drift:
        out.append("- `%s`: `%s` says %s (\"%s\"), the trace says %s (%+.0f%%)" % (
            d["target"], d["doc"], fmt_params(d["claimed"]), d["phrase"],
            fmt_params(d["traced"]), d["offBy"]))
    return "\n".join(out)


# ---------------------------------------------------------------- the whole body

def assemble_report(entries: List[Dict[str, Any]], version: str, previous: Optional[Dict[str, Any]] = None,
                    drift: Optional[List[Dict[str, Any]]] = None, commit: Optional[str] = None,
                    repo: Optional[str] = None, run_url: Optional[str] = None,
                    config_name: str = ".neurarch.yml", generated: Optional[str] = None,
                    api_host: str = "www.neurarch.com") -> str:
    """The whole issue body. The marker is the first line and the data block is the last."""
    rows = [model_row(e) for e in entries]
    generated = generated or _now_iso()
    where = "`%s`" % commit[:7] if commit else "the default branch"
    if repo and commit:
        where = "[`%s`](https://github.com/%s/commit/%s)" % (commit[:7], repo, commit)
    parts = [
        MARKER,
        "## %s" % TITLE,
        "",
        "Every model listed in `%s`, traced at %s on %s. This is a record, not a gate: "
        "report mode never fails the job." % (config_name, where, generated.split("T")[0]),
        "",
        render_table(rows),
        "",
        render_findings(entries),
        "",
        render_changes(previous, rows),
        "",
        render_drift(drift or []),
        "",
    ]
    footer = ("<sub>neurarch-bot v%s, weekly report. The traced graph (layer structure and shapes, "
              "never weights or data) was sent to %s to compute these plans; nothing is shared publicly. "
              "[How it works](%s)" % (version, api_host, BOT_URL))
    if run_url:
        footer += " · [run log](%s)" % run_url
    footer += "</sub>"
    parts.append(footer)
    parts.append("")
    parts.append(embed_data(rows, commit=commit, generated=generated))
    return "\n".join(parts) + "\n"


def change_comment(previous: Optional[Dict[str, Any]], entries: List[Dict[str, Any]],
                   drift: Optional[List[Dict[str, Any]]] = None, commit: Optional[str] = None,
                   run_url: Optional[str] = None) -> str:
    """The short note added to the issue when the body is updated, naming what moved."""
    rows = [model_row(e) for e in entries]
    at = " at `%s`" % commit[:7] if commit else ""
    lines = diff_reports(previous, rows) if previous else []
    head = "Report refreshed%s." % at
    if previous is None:
        head = "First report%s: %d model%s traced." % (at, len(rows), "" if len(rows) == 1 else "s")
        body = [head]
    elif lines:
        body = ["%s Changed since %s:" % (head, _pretty_date(previous.get("generated"))), ""]
        body += ["- %s" % line for line in lines[:10]]
        if len(lines) > 10:
            body.append("- and %d more, in the body above." % (len(lines) - 10))
    else:
        body = ["%s Nothing changed since %s." % (head, _pretty_date(previous.get("generated")))]
    blockers = sum(len(_findings(e, "blockers")) for e in entries)
    if blockers:
        body.append("")
        body.append("%d blocker%s stand%s in the report above; report mode does not fail the job."
                    % (blockers, "" if blockers == 1 else "s", "s" if blockers == 1 else ""))
    if drift:
        body.append("")
        body.append("%d documentation drift finding%s." % (len(drift), "" if len(drift) == 1 else "s"))
    if run_url:
        body.append("")
        body.append("<sub>[run log](%s)</sub>" % run_url)
    return "\n".join(body) + "\n"
