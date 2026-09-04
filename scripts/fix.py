"""The fix half of neurarch-bot: turn a blocked plan into a pull request, or say why not.

The rule this file exists to hold: **nothing is pushed that the bot's own gate has not
accepted.** A patch is applied to a scratch worktree, the model is traced again and planned
again, and only a plan that is legal AND within `PARAM_TOLERANCE` of the pull request's own
head is allowed anywhere near a branch. A "fix" that deletes half the model turns every
blocker green and is not a fix, which is what the parameter guard is for.

Two sources of patches, tried in order, because they do not cost the same:

1. **Exact** - `suggest_fix` in neurarch-mcp. It reads the .py, lints it, and returns a
   unified diff for the rules that pin a number or an order. No model call, no key, no
   metering. There is no one-shot `suggest_fix` subcommand on the CLI (only `lint` and
   `check`), so this speaks the server's stdio JSON-RPC directly: two writes and two reads,
   standard library only, no MCP client library and no framework. `mcp_suggest_fix` is the
   whole of it.
2. **Proposed** - `POST /api/v1/fix`. Metered against the API key owner's agent quota, so it
   is skipped outright when no `api-key` is set, and the comment says so rather than
   pretending the source was tried.

Everything here is a pure function over data except `mcp_suggest_fix` (a subprocess),
`request_fix` (one HTTP POST) and the small git helpers, all of which take their side effects
through injected callables so the tests need no network, no torch and no GitHub.
"""
import json
import os
import re
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

FIX_PATH = "/api/v1/fix"
BRANCH_PREFIX = "neurarch/fix-pr"
MAX_ROUNDS = 3
PARAM_TOLERANCE = 0.10
MAX_FIX_FILES = 8
MCP_PACKAGE = os.environ.get("NEURARCH_MCP_PACKAGE") or "neurarch-mcp@0.15"
MCP_TIMEOUT = 300
TIMEOUT_SECONDS = 120


SKIPPED_NO_KEY = ("the server source (`POST /api/v1/fix`) was skipped: it is metered against the "
                  "API key owner's agent quota and no `api-key` input is set")


class FixError(Exception):
    """A fix source could not be reached; the message is one sentence a reader can act on."""


# ---------------------------------------------------------------- names and numbers

def branch_name(pr: int) -> str:
    return "%s%d" % (BRANCH_PREFIX, int(pr))


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


def params_delta_pct(new: Optional[float], reference: Optional[float]) -> Optional[float]:
    if new is None or reference is None or not reference:
        return None
    return 100.0 * (float(new) - float(reference)) / float(reference)


def within_tolerance(new: Optional[float], reference: Optional[float],
                     tolerance: float = PARAM_TOLERANCE) -> bool:
    """The parameter guard: a fix may not move the model more than `tolerance` off the head.

    An unknown count on either side is NOT within tolerance. A guard that passes when it
    cannot measure is not a guard, and this one stands between a model and a git push.
    """
    pct = params_delta_pct(new, reference)
    if pct is None:
        return False
    return abs(pct) <= tolerance * 100.0 + 1e-9


# ---------------------------------------------------------------- patches

_HEADER = re.compile(r"^(---|\+\+\+)\s+(\S+)")


def normalize_diff(diff: str, path: str) -> str:
    """Rewrite the `---`/`+++` headers to exactly `path`, so every diff applies with -p0.

    Sources disagree about the header convention: suggest_fix writes `a/small_cnn.py` from the
    file's basename, a model asked for a diff may write `b/models/small_cnn.py` or no prefix
    at all. The path we apply to is the one the caller vouched for, never the one in the text.
    """
    out = []
    seen = 0
    for line in diff.split("\n"):
        m = _HEADER.match(line)
        if m and seen < 2:
            out.append("%s %s" % (m.group(1), path))
            seen += 1
            continue
        out.append(line)
    text = "\n".join(out)
    return text if text.endswith("\n") else text + "\n"


def patch_paths_allowed(patches: Sequence[Dict[str, Any]], allowed: Sequence[str]) -> Tuple[bool, str]:
    """Refuse a patch against anything but the files the config named.

    The bot has a write token. The blast radius of "apply whatever came back" is the whole
    repository, so the allowlist is the config's own model files and nothing else.
    """
    allow = set(os.path.normpath(p) for p in allowed)
    for p in patches:
        path = str(p.get("path") or "")
        norm = os.path.normpath(path)
        if not path or path.startswith("/") or ".." in path.split("/"):
            return False, "patch path %r is not a relative path inside the repository" % path
        if norm not in allow:
            return False, "patch targets `%s`, which `.neurarch.yml` does not list (allowed: %s)" % (
                path, ", ".join(sorted(allow)) or "none")
    return True, ""


def apply_patches(patches: Sequence[Dict[str, Any]], workdir: str,
                  runner: Optional[Callable[[List[str], str, Optional[str]], Tuple[int, str]]] = None
                  ) -> Tuple[bool, str]:
    """Apply every patch in `workdir` with `git apply -p0`. All or nothing."""
    run = runner or _git_apply
    for p in patches:
        path = str(p.get("path") or "")
        diff = normalize_diff(str(p.get("diff") or ""), path)
        if not diff.strip():
            return False, "the patch for `%s` is empty" % path
        code, err = run(["git", "apply", "-p0", "--whitespace=nowarn", "-"], workdir, diff)
        if code != 0:
            return False, "the patch for `%s` did not apply: %s" % (path, (err or "").strip().splitlines()[-1:] or "")
    return True, ""


def _git_apply(cmd: List[str], cwd: str, stdin: Optional[str]) -> Tuple[int, str]:
    r = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True)
    return r.returncode, r.stderr or r.stdout


# ---------------------------------------------------------------- source 1: neurarch-mcp

def mcp_suggest_fix(py_path: str, cwd: str, timeout: int = MCP_TIMEOUT,
                    package: Optional[str] = None) -> Dict[str, Any]:
    """Call `suggest_fix` on one .py over the MCP server's stdio JSON-RPC.

    No MCP client library: the transport is newline-delimited JSON-RPC on stdin/stdout, which
    is two writes and two reads.  `severity: info` asks for every finding rather than the
    default floor, because the gate downstream re-verifies anyway and a finding we refused to
    even look at cannot be fixed.

    The answer is read as it arrives and the server is killed the moment it lands. Waiting for
    the process to exit instead cost a whole run: a stdio server is allowed to sit there after
    it has answered, and `communicate()` waits for EOF, so a call that had already succeeded
    was reported as a timeout.
    """
    cmd = ["npx", "-y", package or MCP_PACKAGE, py_path, "--tools=full"]
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
    except (OSError, ValueError) as e:
        raise FixError("could not start neurarch-mcp (%s); is node on the runner?" % e)

    lines: List[str] = []
    answer: Dict[str, Any] = {}
    done = threading.Event()

    def pump():
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                lines.append(line)
                found = _mcp_result(line, 2)
                if found is not None:
                    answer.update(found)
                    done.set()
                    return
                if _is_error_for(line, 2):
                    done.set()
                    return
        finally:
            done.set()

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        proc.stdin.write(_mcp_requests())  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]
    except (BrokenPipeError, OSError) as e:
        proc.kill()
        raise FixError("neurarch-mcp closed its input (%s)" % e)
    finished = done.wait(timeout)
    proc.kill()
    if not finished:
        raise FixError("neurarch-mcp did not answer within %ds" % timeout)
    if not answer:
        tail = "\n".join("".join(lines).rstrip().splitlines()[-3:])
        raise FixError("neurarch-mcp returned no suggest_fix result (%s)" % (tail or "no output"))
    return answer


def _is_error_for(line: str, want_id: int) -> bool:
    """A JSON-RPC error for our call: stop waiting, there is no answer coming."""
    line = line.strip()
    if not line.startswith("{"):
        return False
    try:
        msg = json.loads(line)
    except ValueError:
        return False
    return msg.get("id") == want_id and "error" in msg


def _mcp_requests() -> str:
    return "\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "neurarch-bot", "version": "0.3.0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "suggest_fix", "arguments": {"severity": "info"}}},
    ]) + "\n"


def _mcp_result(stdout: str, want_id: int) -> Optional[Dict[str, Any]]:
    """The response with `id == want_id`, unwrapped from MCP's content envelope."""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("id") != want_id:
            continue
        result = msg.get("result")
        if not isinstance(result, dict):
            return None
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        for part in result.get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                try:
                    parsed = json.loads(part["text"])
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        return None
    return None


def exact_patches(result: Dict[str, Any], path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Turn one suggest_fix result into patches against `path`, plus the notes it refused on."""
    patches: List[Dict[str, Any]] = []
    for fix in result.get("fixes") or []:
        if not isinstance(fix, dict) or not str(fix.get("diff") or "").strip():
            continue
        patches.append({
            "path": path,
            "diff": normalize_diff(str(fix["diff"]), path),
            "findingTitles": [str(fix.get("summary") or fix.get("rule") or "fix")],
            "rule": fix.get("rule"),
            "confidence": fix.get("confidence"),
        })
    notes = []
    for nf in result.get("notFixable") or []:
        if isinstance(nf, dict) and nf.get("rule"):
            notes.append("%s: %s" % (nf["rule"], nf.get("reason") or "no mechanical fix"))
    return patches, notes


# ---------------------------------------------------------------- source 2: /api/v1/fix

def request_fix(model: Optional[Dict[str, Any]], findings: List[Dict[str, Any]],
                files: List[Dict[str, str]], api: str, api_key: str,
                policy: Optional[Dict[str, Any]] = None,
                timeout: float = TIMEOUT_SECONDS) -> Dict[str, Any]:
    """POST `{model?, findings, files, policy?}` and return `{patches, rationale, model}`."""
    api = api.rstrip("/")
    body: Dict[str, Any] = {"findings": findings[:12], "files": files[:MAX_FIX_FILES]}
    if model is not None:
        body["model"] = model
    if policy:
        body["policy"] = policy
    req = urllib.request.Request(
        api + FIX_PATH, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "Authorization": "Bearer " + api_key, "User-Agent": "neurarch-bot"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        raise FixError("%s returned HTTP %d: %s" % (api, e.code, _server_message(raw, e.reason or "error")))
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        raise FixError("could not reach %s (%s)" % (api, getattr(e, "reason", None) or e))
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise FixError("%s returned a response that is not JSON" % api)
    if not isinstance(data, dict) or not isinstance(data.get("patches"), list):
        raise FixError("%s returned a response without patches" % api)
    return data


def _server_message(body: bytes, fallback: str) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return fallback
    if isinstance(data, dict) and isinstance(data.get("error"), str) and data["error"]:
        return data["error"]
    return fallback


# ---------------------------------------------------------------- the rounds

def patch_signature(patches: Sequence[Dict[str, Any]]) -> str:
    """A stable key for a patch set, so a source that keeps proposing the same thing is asked once."""
    return "\n".join(sorted("%s\x00%s" % (p.get("path"), p.get("diff")) for p in patches))


def run_fix_rounds(findings: List[Dict[str, Any]], sources: Sequence[Tuple[str, Callable]],
                   apply_fn: Callable[[Sequence[Dict[str, Any]]], Tuple[bool, str]],
                   verify_fn: Callable[[], Dict[str, Any]],
                   reference_params: Optional[float],
                   reset_fn: Optional[Callable[[], None]] = None,
                   max_rounds: int = MAX_ROUNDS,
                   tolerance: float = PARAM_TOLERANCE) -> Dict[str, Any]:
    """Ask each source in turn for patches, apply them, re-verify, and accept only a legal result.

    `sources` are `(name, fn(findings) -> (patches, note))`; a source with nothing to say
    returns no patches and a note, which is what the comment prints when no fix was pushed.
    `verify_fn` re-traces and re-plans the scratch worktree and returns
    `{status, legal, params, blockers, findings, summary}`.

    A round is one attempt: the worktree is reset to the pull request's head, everything kept
    from earlier rounds is re-applied, the new patches go on top, and the whole thing is
    verified as one. Only a round that reduced the blocker count is kept; a round that did not
    is thrown away rather than left to poison the next one. The findings that come back are
    what the next round's sources are asked about, which is the only reason a second round
    can do better than the first.

    A source that proposes a patch set already tried is skipped, so `max_rounds` bounds real
    work and not the same diff three times.

    Returns `{accepted, source, rounds, patches, reason, attempts, after, notes}`.
    """
    attempts: List[Dict[str, Any]] = []
    remaining = list(findings)
    notes: List[str] = []
    kept: List[Dict[str, Any]] = []
    kept_blockers = len(findings)
    tried = set()
    sources_used: List[str] = []
    for rnd in range(1, max_rounds + 1):
        progressed = False
        for name, fn in sources:
            try:
                patches, note = fn(remaining)
            except FixError as e:
                notes.append("%s: %s" % (name, e))
                continue
            if note:
                notes.append("%s: %s" % (name, note))
            if not patches:
                continue
            candidate = kept + list(patches)
            sig = patch_signature(candidate)
            if sig in tried:
                continue
            tried.add(sig)
            progressed = True
            if reset_fn:
                reset_fn()
            ok, err = apply_fn(candidate)
            if not ok:
                attempts.append({"round": rnd, "source": name, "rejected": err})
                notes.append("%s (round %d): %s" % (name, rnd, err))
                continue
            result = verify_fn()
            accepted, reason = judge(result, reference_params, tolerance)
            attempts.append({"round": rnd, "source": name, "result": result,
                             "rejected": None if accepted else reason})
            if accepted:
                sources_used.append(name)
                return {"accepted": True, "source": " + ".join(dict.fromkeys(sources_used)),
                        "rounds": rnd, "patches": candidate, "reason": "", "attempts": attempts,
                        "after": result, "notes": notes}
            notes.append("%s (round %d): %s" % (name, rnd, reason))
            new_blockers = len(result.get("blockers") or [])
            traced = result.get("status", "planned") == "planned"
            if traced and new_blockers < kept_blockers:
                kept, kept_blockers = candidate, new_blockers
                sources_used.append(name)
                found = result.get("findings")
                if isinstance(found, list):
                    remaining = found
        if not progressed:
            break
    if reset_fn:
        reset_fn()
    return {"accepted": False, "source": None, "rounds": len(attempts), "patches": [],
            "reason": _no_fix_reason(notes), "attempts": attempts, "after": None, "notes": notes}


def _no_fix_reason(notes: List[str]) -> str:
    return "; ".join(notes) if notes else "no source produced a patch"


def judge(result: Dict[str, Any], reference_params: Optional[float],
          tolerance: float = PARAM_TOLERANCE) -> Tuple[bool, str]:
    """The gate. Legal plan, and a parameter count within `tolerance` of the PR's own head."""
    if result.get("status") and result["status"] != "planned":
        return False, "the patched model no longer traces (%s)" % (result.get("error") or result["status"])
    if not result.get("legal"):
        blockers = result.get("blockers") or []
        first = blockers[0].get("title") if blockers and isinstance(blockers[0], dict) else None
        return False, "the plan is still blocked%s" % (" (%s)" % first if first else "")
    params = result.get("params")
    if not within_tolerance(params, reference_params, tolerance):
        pct = params_delta_pct(params, reference_params)
        if pct is None:
            return False, ("the parameter count could not be compared against this pull request's head, "
                           "so the %d%% guard could not run" % int(tolerance * 100))
        return False, ("the fix moves the parameter count by %+.1f%% (%s -> %s), past the %d%% "
                       "guard: a fix that resizes the model is a different change"
                       % (pct, fmt_params(reference_params), fmt_params(params), int(tolerance * 100)))
    return True, ""


# ---------------------------------------------------------------- rendering

def plan_line(label: str, summary: Optional[str], params: Optional[float],
              blockers: Optional[List[Dict[str, Any]]]) -> str:
    n = len(blockers or [])
    verdict = "legal" if n == 0 else "%d blocker%s" % (n, "" if n == 1 else "s")
    return "| %s | %s | %s | %s |" % (label, summary or "?", fmt_params(params), verdict)


def fix_pr_body(pr: int, target: str, before: Dict[str, Any], after: Dict[str, Any],
                source: str, patches: Sequence[Dict[str, Any]], rounds: int,
                version: str, notes: Optional[List[str]] = None,
                skipped: Optional[List[str]] = None) -> str:
    """The fix PR's body: both plans side by side, and which source produced the patch."""
    out = [
        "This fixes the blocker `neurarch-bot` found on #%d." % pr,
        "",
        "It is a pull request into that branch, not into the default branch: the fix belongs on "
        "top of the change that broke it.",
        "",
        "## The plan, before and after",
        "",
        "| | Summary | Params | Verdict |",
        "|---|---|---|---|",
        plan_line("Before (#%d)" % pr, before.get("summary"), before.get("params"), before.get("blockers")),
        plan_line("After (this branch)", after.get("summary"), after.get("params"), after.get("blockers")),
        "",
    ]
    pct = params_delta_pct(after.get("params"), before.get("params"))
    if pct is not None:
        out.append("Parameter count moved %+.1f%%, inside the %d%% guard. A patch that resizes the "
                   "model is a different change and is refused." % (pct, int(PARAM_TOLERANCE * 100)))
        out.append("")
    out.append("## Where the patch came from")
    out.append("")
    named = source[0].upper() + source[1:] if source and source[0].isalpha() else source
    out.append("**%s**, accepted on round %d of %d." % (named, rounds, MAX_ROUNDS))
    out.append("")
    for p in patches:
        titles = ", ".join(p.get("findingTitles") or []) or "no title"
        out.append("- `%s` - %s" % (p.get("path"), titles))
    out.append("")
    for s in skipped or []:
        out.append("_%s_" % s)
    if skipped:
        out.append("")
    out.append("## How this was checked")
    out.append("")
    out.append("The patch was applied to a scratch worktree, `%s` was traced again, and the graph "
               "was planned again. Nothing is pushed unless that second plan is legal and the "
               "parameter count is within %d%% of #%d's own head. This branch is what survived "
               "that gate." % (target, int(PARAM_TOLERANCE * 100), pr))
    if notes:
        out.append("")
        out.append("<details><summary>What else was tried</summary>")
        out.append("")
        for n in notes:
            out.append("- %s" % n)
        out.append("")
        out.append("</details>")
    out.append("")
    out.append("<sub>Opened by neurarch-bot v%s. A pull request opened with the default "
               "`GITHUB_TOKEN` does not trigger workflows, so this one has no bot check of its own; "
               "pass a PAT as `github-token` to get one.</sub>" % version)
    return "\n".join(out) + "\n"


def success_line(pr_number: int, pr_url: str, source: str, before_params: Optional[float],
                 after_params: Optional[float], updated: bool = False) -> str:
    pct = params_delta_pct(after_params, before_params)
    move = "" if pct is None else " Params %s -> %s (%+.1f%%)." % (
        fmt_params(before_params), fmt_params(after_params), pct)
    verb = "updated" if updated else "opened"
    return ("**A fix is open: [#%d](%s).** neurarch-bot %s it, re-traced the patched model and "
            "re-planned it, and the new plan is legal.%s Patch source: %s."
            % (pr_number, pr_url, verb, move, source))


def failure_line(reason: str, skipped: Optional[List[str]] = None) -> str:
    reason = reason.rstrip(".")
    if reason and reason[0].isalpha():
        reason = reason[0].upper() + reason[1:]
    parts = ["**No fix was pushed.** %s." % reason]
    for s in skipped or []:
        parts.append(s)
    parts.append("Nothing was pushed: neurarch-bot only opens a fix its own gate accepted "
                 "(a legal plan, and a parameter count within %d%% of this branch's head)."
                 % int(PARAM_TOLERANCE * 100))
    return " ".join(parts)


MARKER = "<!-- neurarch-bot-fix -->"


def comment_with_fix_line(body: str, line: str) -> str:
    """Add (or replace) the one fix line in the plan comment, just above the footer."""
    block = "%s\n%s" % (MARKER, line)
    if MARKER in body:
        lines = body.split("\n")
        out, i = [], 0
        while i < len(lines):
            if lines[i].strip() == MARKER:
                out.append(MARKER)
                out.append(line)
                i += 2
                continue
            out.append(lines[i])
            i += 1
        return "\n".join(out)
    idx = body.rfind("<sub>")
    if idx == -1:
        return body.rstrip("\n") + "\n\n" + block + "\n"
    return body[:idx] + block + "\n\n" + body[idx:]


# ---------------------------------------------------------------- GitHub

def api_base() -> str:
    return (os.environ.get("GITHUB_API_URL") or "https://api.github.com").rstrip("/")


def gh_request(method: str, path: str, token: str, body: Optional[Dict[str, Any]] = None,
               api: Optional[str] = None, timeout: float = TIMEOUT_SECONDS) -> Any:
    url = (api or api_base()).rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "neurarch-bot",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise FixError("GitHub %s %s returned HTTP %d: %s" % (
            method, path, e.code, _server_message(e.read(), e.reason or "error")))
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        raise FixError("could not reach GitHub (%s)" % (getattr(e, "reason", None) or e))
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def existing_fix_pr(open_prs: Sequence[Dict[str, Any]], branch: str,
                    owner: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The open fix PR for this branch, if there is one.

    A second fix pull request for the same original PR is never opened: the branch name is
    derived from the PR number, so pushing it again updates the one that is already open. This
    is the check that turns "open" into "update".
    """
    want = branch.lower()
    for pr in open_prs or []:
        if not isinstance(pr, dict) or (pr.get("state") or "open") != "open":
            continue
        head = pr.get("head") or {}
        ref = str(head.get("ref") or "").lower()
        if ref != want:
            continue
        if owner and str((head.get("repo") or {}).get("full_name") or owner) != owner:
            continue
        return pr
    return None
