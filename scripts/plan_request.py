"""The wire half of neurarch-bot: POST a traced graph to `/api/v1/plan` and build the PR comment.

Standard library only, no torch, so the request body, the key header, the
rate-limit message and the comment assembly can all be tested with a stub
HTTP server and no model.
"""
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import job  # noqa: E402
import policy_lines  # noqa: E402

__version__ = "0.6.0"

DEFAULT_API = "https://www.neurarch.com"
PLAN_PATH = "/api/v1/plan"
MARKER = "<!-- neurarch-bot -->"
TIMEOUT_SECONDS = 60
BOT_URL = "https://neurarch.com/bot"


class PlanRequestError(Exception):
    """The plan could not be fetched; the message is one sentence a reader can act on."""


def api_base(env: Optional[Dict[str, str]] = None) -> str:
    env = os.environ if env is None else env
    return (env.get("NEURARCH_API") or DEFAULT_API).rstrip("/")


def api_host(base: str) -> str:
    return urlsplit(base).netloc or base


def build_source(target: str, input_spec: str, pr: Optional[int], version: str = __version__) -> Dict[str, Any]:
    source: Dict[str, Any] = {
        "kind": "bot",
        "target": target,
        "input": input_spec,
        "tool": "neurarch-bot/%s" % version,
    }
    if pr is not None:
        source["pr"] = int(pr)
    return source


def _server_message(body: bytes, fallback: str) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return fallback
    if isinstance(data, dict) and isinstance(data.get("error"), str) and data["error"]:
        return data["error"]
    return fallback


def request_plan(
    model: Dict[str, Any],
    base: Optional[Dict[str, Any]],
    source: Dict[str, Any],
    api: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = TIMEOUT_SECONDS,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """POST `{model, base?, policy?, share: false, source}` and return the decoded response.

    Nothing is ever shared publicly from the bot: `share` is pinned to false. `policy` is the
    organisation's own `.neurarch.yml` lines; it needs no API key, because a policy can only
    make the verdict stricter and never more permissive.
    """
    api = (api or api_base()).rstrip("/")
    host = api_host(api)
    body: Dict[str, Any] = {"model": model, "share": False, "source": source}
    if base is not None:
        body["base"] = base
    if policy:
        body["policy"] = policy
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": source.get("tool", "neurarch-bot"),
    }
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(
        api + PLAN_PATH, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        if e.code == 429:
            msg = _server_message(raw, "too many requests")
            raise PlanRequestError(
                "%s rate-limited this request: %s. Keyless calls are limited per hour; "
                "set the `api-key` input (a Neurarch API key, nrk_...) to raise the limit." % (host, msg))
        raise PlanRequestError("%s returned HTTP %d: %s" % (host, e.code, _server_message(raw, e.reason or "error")))
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", None) or e
        raise PlanRequestError("could not reach %s (%s)" % (api, reason))
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise PlanRequestError("%s returned a response that is not JSON" % host)
    if not isinstance(data, dict) or not isinstance(data.get("plan"), dict) \
            or not isinstance(data.get("markdown"), str):
        raise PlanRequestError("%s returned a response without a plan" % host)
    return data


def write_plan(response: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    """Write plan.json, plan.md and plan.txt into `out_dir`; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "json": os.path.join(out_dir, "plan.json"),
        "md": os.path.join(out_dir, "plan.md"),
        "txt": os.path.join(out_dir, "plan.txt"),
    }
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(response.get("plan"), f, indent=2)
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(response.get("markdown", ""))
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write(response.get("text", ""))
    return paths


def has_blocker(plan: Dict[str, Any]) -> bool:
    run = plan.get("run") if isinstance(plan, dict) else None
    if not isinstance(run, dict):
        return False
    if run.get("legal") is False:
        return True
    blockers = run.get("blockers")
    return isinstance(blockers, list) and len(blockers) > 0


def _strip_leading_summary(markdown: str, summary: str) -> str:
    """The server's markdown opens with the bold summary; the section prints it once."""
    lines = markdown.lstrip("\n").split("\n")
    if lines and summary and lines[0].strip() == "**%s**" % summary:
        return "\n".join(lines[1:]).lstrip("\n")
    return markdown


def _badge(entry: Dict[str, Any]) -> str:
    status = entry.get("status")
    if status == "planned":
        plan = entry.get("plan") or {}
        if has_blocker(plan):
            return ":no_entry: blocked"
        warnings = (plan.get("run") or {}).get("warnings") or []
        return ":warning: %d warning%s" % (len(warnings), "" if len(warnings) == 1 else "s") if warnings else ":white_check_mark: will run"
    if status == "trace_failed":
        return ":no_entry: could not trace"
    return ":warning: no plan"


def _fmt_params(n: Any) -> str:
    if not isinstance(n, (int, float)):
        return "?"
    n = float(n)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return "%.1f%s" % (n / div, suffix)
    return "%d" % int(n)


def _finding_line(f: Dict[str, Any], with_fix: bool = True) -> str:
    line = "- %s" % (f.get("title") or "finding")
    if f.get("detail"):
        line += ": %s" % f["detail"]
    if with_fix and f.get("fix"):
        line += " Fix: %s" % f["fix"]
    return line


def render_structural(plan: Dict[str, Any]) -> List[str]:
    """The second block: will it run, what blocks or warns, the house rules, what changed.

    Rendered from the plan JSON (the public contract of `/api/v1/plan`) rather than from the
    server's markdown, because that markdown opens with a memory table sized at the server's
    default batch, which would sit under the job's own number and disagree with it.
    """
    run = plan.get("run") or {}
    model = plan.get("model") or {}
    diff = plan.get("diff") if isinstance(plan.get("diff"), dict) else None
    blockers = [f for f in run.get("blockers") or [] if isinstance(f, dict)]
    warnings = [f for f in run.get("warnings") or [] if isinstance(f, dict)]
    legal = run.get("legal")
    if legal is None:
        verdict = "**cannot say**: %s" % (run.get("withheld") or "the read was incomplete")
    elif legal and not blockers:
        verdict = "yes, %d warning%s" % (len(warnings), "" if len(warnings) == 1 else "s")
    else:
        verdict = "**no**, %d blocker%s" % (len(blockers), "" if len(blockers) == 1 else "s")
    if diff and diff.get("legalityChanged") == "became_illegal":
        verdict += " (**base ran, this does not**)"
    elif diff and diff.get("legalityChanged") == "became_legal":
        verdict += " (base did not run, this does)"
    params = "%s params" % _fmt_params(model.get("params"))
    if diff and isinstance(diff.get("paramsDelta"), (int, float)):
        d = diff["paramsDelta"]
        params += " (%s vs base)" % ("no change" if d == 0 else "%s%s" % ("+" if d > 0 else "-", _fmt_params(abs(d))))
    out = ["**Structural checks.** Will it run: %s. %s layers, %s." % (
        verdict, model.get("layers", "?"), params)]
    if blockers:
        out += ["", "**Blockers**"] + [_finding_line(f) for f in blockers[:6]]
        if len(blockers) > 6:
            out.append("- and %d more" % (len(blockers) - 6))
    if warnings:
        out += ["", "**Warnings**"] + [_finding_line(f, with_fix=False) for f in warnings[:6]]
        if len(warnings) > 6:
            out.append("- and %d more" % (len(warnings) - 6))
    if (plan.get("policy") or {}).get("applied"):
        lines = [f for f in blockers + warnings if f.get("stage") == "policy"]
        out += ["", "**Policy**"]
        out += ["- %s" % f.get("title") + (" (`%s`)" % f["detail"] if f.get("detail") else "")
                for f in lines[:8]] or ["- every policy line passes"]
    if diff and diff.get("lines"):
        lines = list(diff["lines"])
        out += ["", "**Changes vs %s** (+%s -%s ~%s, %s downstream layer%s)" % (
            (diff.get("base") or {}).get("name", "base"), diff.get("added", 0), diff.get("removed", 0),
            diff.get("changed", 0), diff.get("blastRadius", 0), "" if diff.get("blastRadius") == 1 else "s"),
            "```"] + lines[:12] + (["... %d more" % (len(lines) - 12)] if len(lines) > 12 else []) + ["```"]
    return out


def render_entry(entry: Dict[str, Any]) -> str:
    """One model's section: heading, memory against the reserved card, then the structural checks."""
    target = entry.get("target", "?")
    status = entry.get("status")
    head_badge = _badge(entry)
    mem_badge = job.badge(entry.get("sizing")) if status == "planned" else None
    out = ["### `%s` %s" % (target, "%s · %s" % (mem_badge, head_badge) if mem_badge else head_badge), ""]
    if status == "planned":
        plan = entry.get("plan") or {}
        if entry.get("base_note"):
            out.append("_%s_" % entry["base_note"])
            out.append("")
        lead = job.render_lead(entry.get("sizing"), entry.get("job"), entry.get("job_problems"))
        if lead and (entry.get("sizing") or {}).get("ok"):
            out.extend(lead)
            out.append("")
            out.extend(render_structural(plan))
        else:
            # No sizing on this runner: the server's own plan, as before, under whatever
            # line says why the memory block is missing.
            if lead:
                out.extend(lead)
                out.append("")
            summary = plan.get("summary") or "plan for %s" % target
            out.append("**%s**" % summary)
            out.append("")
            out.append(_strip_leading_summary(entry.get("markdown", ""), summary).rstrip())
        # The findings name the rule; this names the line somebody wrote it on. A policy
        # blocker with nowhere to go reads as our opinion rather than as the repository's.
        section = policy_lines.render_section(entry.get("rule_lines") or [])
        if section:
            out.append("")
            out.append(section)
        return "\n".join(out)

    if status == "trace_failed":
        out.append("**Could not trace `%s` with input `%s`; the entry point in `.neurarch.yml` needs fixing "
                   "(or the model does not construct).**" % (target, entry.get("input", "")))
        out.append("")
        out.append("```")
        out.append((entry.get("error") or "no output").rstrip())
        out.append("```")
    else:
        out.append("**The model traced but no plan came back: %s**" % (entry.get("error") or "unknown error"))
    lint_md = entry.get("lint_markdown")
    if lint_md:
        out.append("")
        out.append("Static read of the file instead (no import, no forward pass):")
        out.append("")
        out.append(lint_md.rstrip())
    out.append("")
    out.append("_The traceback is what a trace failure looks like; this entry did not fail the job by itself. "
               "A blocker found by the static read does._")
    return "\n".join(out)


def assemble_comment(entries: List[Dict[str, Any]], version: str = __version__,
                     run_url: Optional[str] = None, note: Optional[str] = None) -> str:
    """One comment for every affected model. The marker is the first line so the bot can find it again."""
    parts = [MARKER, "## Neurarch plan", ""]
    if note:
        parts.append(note)
        parts.append("")
    for entry in entries:
        parts.append(render_entry(entry))
        parts.append("")
    footer = ("<sub>neurarch-bot v%s. The traced graph (layer structure and shapes, never weights or data) "
              "was sent to %s for the structural checks; nothing is shared publicly. "
              % (version, api_host(api_base())))
    if any((e.get("sizing") or {}).get("ok") for e in entries):
        vendored = job.bundle_version()
        footer += ("Memory, batch and cost are estimates (AdamW, bf16 autocast) computed on the runner "
                   "by the plan bundle vendored in neurarch-bot%s. "
                   % (", neurarch `%s`" % vendored[:7] if vendored else ""))
    footer += "[How it works](%s)" % BOT_URL
    if run_url:
        footer += " · [run log](%s)" % run_url
    footer += "</sub>"
    parts.append(footer)
    return "\n".join(parts) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    """`plan_request.py --model head.json [--base base.json] --target T --input I [--pr N] --out DIR`."""
    import argparse
    p = argparse.ArgumentParser(prog="plan_request.py")
    p.add_argument("--model", required=True)
    p.add_argument("--base", default=None)
    p.add_argument("--target", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--pr", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--api", default=None)
    args = p.parse_args(argv)
    with open(args.model, "r", encoding="utf-8") as f:
        model = json.load(f)
    base = None
    if args.base:
        with open(args.base, "r", encoding="utf-8") as f:
            base = json.load(f)
    try:
        response = request_plan(model, base, build_source(args.target, args.input, args.pr),
                                api=args.api, api_key=os.environ.get("NEURARCH_API_KEY") or None)
    except PlanRequestError as e:
        print("plan_request: %s" % e, file=sys.stderr)
        return 1
    paths = write_plan(response, args.out)
    print(response.get("text", ""))
    print("wrote %s" % ", ".join(paths.values()), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
