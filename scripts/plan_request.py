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

__version__ = "0.3.0"

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


def render_entry(entry: Dict[str, Any]) -> str:
    """One model's section: heading, bold summary, then the server's markdown (or the fallback)."""
    target = entry.get("target", "?")
    out = ["### `%s` %s" % (target, _badge(entry)), ""]
    status = entry.get("status")
    if status == "planned":
        plan = entry.get("plan") or {}
        summary = plan.get("summary") or "plan for %s" % target
        out.append("**%s**" % summary)
        out.append("")
        if entry.get("base_note"):
            out.append("_%s_" % entry["base_note"])
            out.append("")
        out.append(_strip_leading_summary(entry.get("markdown", ""), summary).rstrip())
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
              "was sent to %s to compute this plan; nothing is shared publicly. "
              "[How it works](%s)" % (version, api_host(api_base()), BOT_URL))
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
