"""neurarch-bot driver: read `.neurarch.yml`, trace the models, ask `/api/v1/plan` for each,
and write one comment (or one issue) body plus the step outputs.

Two modes over the same code path. `pr` finds the models this pull request touches, traces HEAD
and BASE, and writes a comment that a blocker turns red. `report` traces every listed model at
the current commit with no base, and writes the weekly issue body, which never fails the job.
Both walk `process_model`, so the config parsing, the trace and the plan request are one path.

Runs inside the composite action after torch and neurarch-trace are installed. Everything that
leaves the machine goes through `plan_request.request_plan`; this file never reads a weight.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix as fixmod  # noqa: E402
import plan_request  # noqa: E402
import report  # noqa: E402

__version__ = plan_request.__version__
LINT_URL = "https://raw.githubusercontent.com/neurarch-ai/neurarch-lint/v1/lint.mjs"
TRACE_TIMEOUT = 900


def log(msg: str) -> None:
    print("neurarch-bot: %s" % msg, flush=True)


def env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def run(cmd: List[str], cwd: Optional[str] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def git(args: List[str], cwd: str) -> subprocess.CompletedProcess:
    return run(["git"] + args, cwd=cwd)


# ---------------------------------------------------------------- config

def load_config(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit("neurarch-bot: %s must be a mapping with a `models:` list" % path)
    return data


def parse_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise SystemExit("neurarch-bot: `models:` must be a non-empty list")
    out = []
    for i, m in enumerate(models):
        if not isinstance(m, dict) or not isinstance(m.get("path"), str) or ":" not in m["path"]:
            raise SystemExit("neurarch-bot: models[%d] needs `path: file.py:ClassOrFactory`" % i)
        if "input" not in m:
            raise SystemExit("neurarch-bot: models[%d] (%s) needs `input: 1,3,224,224` (add `:long` for token ids)"
                             % (i, m["path"]))
        file_part = m["path"].rsplit(":", 1)[0]
        extra = m.get("extra_inputs") or []
        if not isinstance(extra, list):
            raise SystemExit("neurarch-bot: models[%d].extra_inputs must be a list" % i)
        out.append({
            "target": m["path"],
            "file": os.path.normpath(file_part),
            "input": str(m["input"]),
            "extra_inputs": [str(x) for x in extra],
            "depth": m.get("depth"),
            "name": m.get("name"),
            "policy": m.get("policy") if isinstance(m.get("policy"), dict) else None,
        })
    return out


def parse_policy(config: Dict[str, Any], model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The policy that applies to one model: the top-level block, then the model's own on top.

    A repository has one set of house rules and, sometimes, a tighter line for one model (a
    500K cap on the small classifier that a 10M transformer in the same config would fail by
    construction). Merging per key rather than replacing wholesale is what makes the top-level
    block usable at all: an entry that tightens one line does not silently drop the rest.
    """
    merged: Dict[str, Any] = {}
    top = config.get("policy")
    if isinstance(top, dict):
        merged.update(top)
    own = model.get("policy")
    if isinstance(own, dict):
        merged.update(own)
    return merged or None


def is_affected(model: Dict[str, Any], changed: List[str], listed_files: Optional[List[str]] = None) -> bool:
    """A listed file changed, or anything under its directory did (a helper module it imports).

    A sibling that is itself a listed model does not count as a helper: editing `models/cnn.py`
    is a change to the CNN entry, not to every model that shares the folder.
    """
    file = model["file"]
    directory = os.path.dirname(file)
    others = set(os.path.normpath(f) for f in (listed_files or []) if os.path.normpath(f) != file)
    for c in changed:
        c = os.path.normpath(c)
        if c == file:
            return True
        if c in others:
            continue
        if directory and (c == directory or c.startswith(directory + os.sep)):
            return True
    return False


# ---------------------------------------------------------------- git

def changed_files(workspace: str, base_sha: str, head_sha: str) -> List[str]:
    """Files the PR changes: merge-base..head. Falls back to base..head, fetching the shas if absent."""
    for sha in (base_sha, head_sha):
        if git(["cat-file", "-e", sha + "^{commit}"], workspace).returncode != 0:
            log("fetching %s (checkout was shallow?)" % sha[:10])
            git(["fetch", "--no-tags", "origin", sha], workspace)
    mb = git(["merge-base", base_sha, head_sha], workspace)
    start = mb.stdout.strip() if mb.returncode == 0 and mb.stdout.strip() else base_sha
    r = git(["diff", "--name-only", start, head_sha], workspace)
    if r.returncode != 0:
        log("git diff failed: %s" % r.stderr.strip())
        return []
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def add_base_worktree(workspace: str, base_sha: str, tmp: str) -> Optional[str]:
    path = os.path.join(tmp, "neurarch-base")
    if os.path.isdir(path):
        git(["worktree", "remove", "--force", path], workspace)
        shutil.rmtree(path, ignore_errors=True)
    r = git(["worktree", "add", "--detach", path, base_sha], workspace)
    if r.returncode != 0:
        log("could not check out base %s: %s" % (base_sha[:10], r.stderr.strip()))
        return None
    return path


def remove_worktree(workspace: str, path: Optional[str]) -> None:
    if path:
        git(["worktree", "remove", "--force", path], workspace)


# ---------------------------------------------------------------- trace

def trace(model: Dict[str, Any], cwd: str, out_path: str) -> Tuple[bool, str]:
    cmd = [sys.executable, "-m", "neurarch_trace", model["target"], "--input", model["input"]]
    for extra in model["extra_inputs"]:
        cmd += ["--input", extra]
    if model.get("depth") is not None:
        cmd += ["--depth", str(model["depth"])]
    if model.get("name"):
        cmd += ["--name", str(model["name"])]
    cmd += ["-o", out_path, "--verbose"]
    try:
        r = run(cmd, cwd=cwd, timeout=TRACE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, "neurarch-trace timed out after %ds" % TRACE_TIMEOUT
    if r.returncode == 0 and os.path.isfile(out_path):
        return True, r.stdout.strip()
    return False, tail(r.stderr or r.stdout, 20)


def tail(text: str, n: int) -> str:
    lines = [l for l in text.rstrip().splitlines() if "UserWarning: Failed to initialize NumPy" not in l
             and "_conversion_method_template" not in l]
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------- static fallback

def lint_script(tmp: str) -> Optional[str]:
    local = os.environ.get("NEURARCH_LINT_MJS")
    if local and os.path.isfile(local):
        return local
    path = os.path.join(tmp, "neurarch-lint.mjs")
    if os.path.isfile(path):
        return path
    try:
        with urllib.request.urlopen(LINT_URL, timeout=30) as resp, open(path, "wb") as f:
            f.write(resp.read())
        return path
    except Exception as e:  # noqa: BLE001
        log("could not fetch neurarch-lint for the static fallback: %s" % e)
        return None


def static_lint(file: str, workspace: str, tmp: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    script = lint_script(tmp)
    if not script or not shutil.which("node"):
        return None, []
    md = run(["node", script, "--markdown", file], cwd=workspace)
    js = run(["node", script, "--json", file], cwd=workspace)
    findings: List[Dict[str, Any]] = []
    try:
        findings = json.loads(js.stdout).get("findings") or []
    except ValueError:
        pass
    return (md.stdout.strip() or None), findings


# ---------------------------------------------------------------- main

def slug(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", target).strip("-")


def process_model(model: Dict[str, Any], workspace: str, base_dir: Optional[str], tmp: str,
                  pr: Optional[int], api_key: Optional[str],
                  policy: Optional[Dict[str, Any]] = None,
                  out_root: Optional[str] = None) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"target": model["target"], "input": model["input"], "status": "planned",
                             "blockers": 0, "warnings": 0, "legal": True}
    out_dir = os.path.join(out_root or os.path.join(tmp, "neurarch-bot"), slug(model["target"]))
    os.makedirs(out_dir, exist_ok=True)
    head_path = os.path.join(out_dir, "head.neurarch.json")
    log("tracing %s (input %s)" % (model["target"], model["input"]))
    ok, msg = trace(model, workspace, head_path)
    if not ok:
        log("trace failed for %s" % model["target"])
        entry.update({"status": "trace_failed", "error": msg})
        lint_md, findings = static_lint(model["file"], workspace, tmp)
        entry["lint_markdown"] = lint_md
        entry["lint_findings"] = findings
        entry["blockers"] = sum(1 for f in findings if f.get("severity") == "block")
        entry["warnings"] = sum(1 for f in findings if f.get("severity") == "warn")
        entry["legal"] = entry["blockers"] == 0
        return entry

    base_graph = None
    if base_dir is not None:
        if os.path.isfile(os.path.join(base_dir, model["file"])):
            base_path = os.path.join(out_dir, "base.neurarch.json")
            log("tracing base %s" % model["target"])
            bok, bmsg = trace(model, base_dir, base_path)
            if bok:
                with open(base_path, "r", encoding="utf-8") as f:
                    base_graph = json.load(f)
            else:
                entry["base_note"] = "The base version did not trace, so there is no diff: %s" % (
                    bmsg.splitlines()[-1] if bmsg else "unknown error")
        else:
            entry["base_note"] = "New model: `%s` does not exist on the base branch, so there is no diff." % model["file"]
    with open(head_path, "r", encoding="utf-8") as f:
        head_graph = json.load(f)

    source = plan_request.build_source(model["target"], " ".join([model["input"]] + model["extra_inputs"]), pr)
    try:
        response = plan_request.request_plan(head_graph, base_graph, source, api_key=api_key,
                                             policy=policy)
    except plan_request.PlanRequestError as e:
        log("plan request failed for %s: %s" % (model["target"], e))
        entry.update({"status": "plan_failed", "error": str(e)})
        lint_md, findings = static_lint(model["file"], workspace, tmp)
        entry["lint_markdown"] = lint_md
        entry["lint_findings"] = findings
        entry["blockers"] = sum(1 for f in findings if f.get("severity") == "block")
        entry["warnings"] = sum(1 for f in findings if f.get("severity") == "warn")
        entry["legal"] = entry["blockers"] == 0
        return entry

    paths = plan_request.write_plan(response, out_dir)
    plan = response["plan"]
    runinfo = plan.get("run") or {}
    entry.update({
        "plan": plan,
        "markdown": response.get("markdown", ""),
        "paths": paths,
        "blockers": len(runinfo.get("blockers") or []),
        "warnings": len(runinfo.get("warnings") or []),
        "legal": not plan_request.has_blocker(plan),
    })
    print(response.get("text", ""), flush=True)
    return entry


def is_clean(entry: Dict[str, Any]) -> bool:
    """Planned, legal, no warnings, and structurally identical to base: nothing to say."""
    if entry.get("status") != "planned" or not entry.get("legal") or entry.get("warnings"):
        return False
    diff = (entry.get("plan") or {}).get("diff")
    if not isinstance(diff, dict):
        return False
    return (diff.get("added", 0) + diff.get("removed", 0) + diff.get("changed", 0)) == 0 \
        and not diff.get("lines")


def summary_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    row = {k: entry.get(k) for k in ("target", "input", "status", "legal", "blockers", "warnings")}
    plan = entry.get("plan")
    if isinstance(plan, dict):
        row["summary"] = plan.get("summary")
        row["params"] = (plan.get("model") or {}).get("params")
        row["fingerprint"] = (plan.get("model") or {}).get("fingerprint")
        cost = plan.get("cost") or {}
        row["estCostUsd"] = cost.get("estCostUsd")
        row["fitsGpu"] = cost.get("fitsGpu")
        diff = plan.get("diff")
        if isinstance(diff, dict):
            row["paramsDelta"] = diff.get("paramsDelta")
            row["costDeltaUsd"] = diff.get("costDeltaUsd")
            row["blastRadius"] = diff.get("blastRadius")
            row["legalityChanged"] = diff.get("legalityChanged")
    if entry.get("error"):
        row["error"] = entry["error"].splitlines()[-1] if entry["error"] else None
    if entry.get("paths"):
        row["plan_json"] = entry["paths"]["json"]
    return row


def write_outputs(values: Dict[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    for k, v in values.items():
        v = str(v)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                if "\n" in v:
                    f.write("%s<<NEURARCH_EOF\n%s\nNEURARCH_EOF\n" % (k, v))
                else:
                    f.write("%s=%s\n" % (k, v))
        else:
            print("output %s=%s" % (k, v if len(v) < 200 else v[:200] + "..."))


def read_previous_report(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Last week's numbers, parsed back out of the issue body this run is about to overwrite."""
    if not path or not os.path.isfile(path):
        log("no previous report on file; this is a first report.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        previous = report.parse_report_data(f.read())
    if previous is None:
        log("the previous report carries no machine-readable block; no week-over-week diff this time.")
    else:
        log("read %d model(s) from the previous report (%s)." % (
            len(previous.get("models") or []), previous.get("generated")))
    return previous


def collect_drift(models: List[Dict[str, Any]], entries: List[Dict[str, Any]], workspace: str) -> List[Dict[str, Any]]:
    """Parameter counts written next to each model that the trace disagrees with."""
    drift: List[Dict[str, Any]] = []
    for model, entry in zip(models, entries):
        params = ((entry.get("plan") or {}).get("model") or {}).get("params")
        others = [m for m in models if m["target"] != model["target"]]
        found = report.doc_drift(model, params, workspace, other_models=others)
        if found:
            log("doc drift: %s says %s params, the trace says %s" % (
                found["doc"], report.fmt_params(found["claimed"]), report.fmt_params(found["traced"])))
            drift.append(found)
    return drift


def run_report(models: List[Dict[str, Any]], workspace: str, tmp: str, api_key: Optional[str],
               config_path: str, run_url: Optional[str], base_outputs: Dict[str, str],
               policies: Optional[Dict[str, Any]] = None) -> int:
    """Report mode: every listed model at this commit, one issue body, no verdict on the job."""
    policies = policies or {}
    previous = read_previous_report(os.environ.get("NEURARCH_PREVIOUS_REPORT"))
    log("planning all %d listed model(s) at the current commit: no base, no diff against a branch."
        % len(models))
    entries = [process_model(m, workspace, None, tmp, None, api_key, policy=policies.get(m["target"]))
               for m in models]
    drift = collect_drift(models, entries, workspace)

    commit = os.environ.get("GITHUB_SHA")
    body = report.assemble_report(
        entries, __version__, previous=previous, drift=drift, commit=commit,
        repo=os.environ.get("GITHUB_REPOSITORY"), run_url=run_url,
        config_name=os.path.basename(config_path),
        api_host=plan_request.api_host(plan_request.api_base()))
    comment = report.change_comment(previous, entries, drift, commit=commit, run_url=run_url)

    body_path = os.path.join(tmp, "neurarch-report-body.md")
    comment_path = os.path.join(tmp, "neurarch-report-comment.md")
    with open(body_path, "w", encoding="utf-8") as f:
        f.write(body)
    with open(comment_path, "w", encoding="utf-8") as f:
        f.write(comment)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(body.replace(report.MARKER + "\n", "", 1))

    blockers = sum(e["blockers"] for e in entries)
    warnings = sum(e["warnings"] for e in entries)
    out = dict(base_outputs)
    out.update({
        "legal": "true" if all(e["legal"] for e in entries) else "false",
        "blockers": str(blockers),
        "warnings": str(warnings),
        "plans-json": json.dumps([summary_row(e) for e in entries], separators=(",", ":")),
        "fail": "false",  # A report is a record, not a gate.
        "comment": "none",
        "affected": str(len(entries)),
        "report": "ready",
        "report-body-path": body_path,
        "report-comment-path": comment_path,
        "report-title": report.TITLE,
    })
    write_outputs(out)
    log("report for %d model(s): %d blocker(s), %d warning(s), %d drift finding(s); the job stays green."
        % (len(entries), blockers, warnings, len(drift)))
    return 0


# ---------------------------------------------------------------- fix stage


def fix_state_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """What the fix stage needs from a planned model, and nothing else.

    The fix runs as a second step so the comment is already posted when it starts; the two
    steps talk through this file rather than through the environment, because a plan's
    blockers do not fit in a step output and a truncated blocker is a fix aimed at nothing.
    """
    plan = entry.get("plan") or {}
    run = plan.get("run") or {}
    findings = list(run.get("blockers") or [])
    if not findings and entry.get("lint_findings"):
        findings = [{"title": f.get("rule") or "finding", "detail": f.get("message"),
                     "stage": "static"} for f in entry["lint_findings"] if f.get("severity") == "block"]
    return {
        "target": entry.get("target"),
        "status": entry.get("status"),
        "legal": entry.get("legal"),
        "blockers": findings,
        "summary": plan.get("summary"),
        "params": (plan.get("model") or {}).get("params"),
    }


def plan_snapshot(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The three numbers the gate and the fix PR body compare on."""
    plan = entry.get("plan") or {}
    run = plan.get("run") or {}
    return {
        "status": entry.get("status"),
        "error": (entry.get("error") or "").splitlines()[-1] if entry.get("error") else None,
        "legal": bool(entry.get("legal")),
        "summary": plan.get("summary"),
        "params": (plan.get("model") or {}).get("params"),
        "blockers": list(run.get("blockers") or []),
        "findings": list(run.get("blockers") or []),
    }


def add_fix_worktree(workspace: str, head_sha: str, tmp: str) -> Optional[str]:
    path = os.path.join(tmp, "neurarch-fix")
    if os.path.isdir(path):
        git(["worktree", "remove", "--force", path], workspace)
        shutil.rmtree(path, ignore_errors=True)
    r = git(["worktree", "add", "--detach", path, head_sha], workspace)
    if r.returncode != 0:
        log("could not check out the pull request head %s: %s" % (head_sha[:10], r.stderr.strip()))
        return None
    return path


def read_files(root: str, paths: List[str]) -> List[Dict[str, str]]:
    out = []
    for rel in paths:
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            continue
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            out.append({"path": rel, "text": f.read()})
    return out


def exact_source(scratch: str, files: List[str]):
    """Source 1: neurarch-mcp `suggest_fix` over stdio JSON-RPC. No model call, no key."""
    def source(_findings):
        patches: List[Dict[str, Any]] = []
        reasons: List[str] = []
        for rel in files:
            if not os.path.isfile(os.path.join(scratch, rel)) or not rel.endswith(".py"):
                continue
            result = fixmod.mcp_suggest_fix(rel, scratch)
            got, notes = fixmod.exact_patches(result, rel)
            patches.extend(got)
            reasons.extend(notes)
        if patches:
            return patches, ""
        return [], ("no mechanical fix for these findings"
                    + (" (%s)" % "; ".join(reasons[:3]) if reasons else ""))
    return source


def server_source(scratch: str, files: List[str], api_key: Optional[str],
                  policy: Optional[Dict[str, Any]], allowed: List[str]):
    """Source 2: `POST /api/v1/fix`. Metered against the key owner's quota, so it is skipped
    outright when there is no key rather than called and refused."""
    def source(findings):
        if not api_key:
            return [], fixmod.SKIPPED_NO_KEY
        payload = read_files(scratch, files)
        if not payload:
            return [], "none of the listed model files are readable"
        data = fixmod.request_fix(None, findings, payload, plan_request.api_base(), api_key,
                                  policy=policy)
        patches = [p for p in data.get("patches") or [] if isinstance(p, dict)]
        ok, why = fixmod.patch_paths_allowed(patches, allowed)
        if not ok:
            return [], "the server proposed a patch this bot will not apply: %s" % why
        if not patches:
            return [], "the server had no patch for these findings (%s)" % (data.get("rationale") or "no rationale")
        for p in patches:
            p.setdefault("findingTitles", [])
        return patches, ""
    return source


def run_fix(workspace: str, tmp: str, api_key: Optional[str]) -> int:
    """Second step of PR mode: try to fix the blockers the first step found, and push only a
    fix this bot's own gate accepted."""
    out = {"fix-status": "none", "fix-line": "", "fix-branch": "", "fix-pr-url": "",
           "fix-pr-number": "", "fix-comment-path": ""}
    state_path = os.environ.get("NEURARCH_STATE") or os.path.join(tmp, "neurarch-bot-state.json")
    token = (os.environ.get("INPUT_GITHUB_TOKEN") or "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    if not os.path.isfile(state_path):
        log("no state from the plan step; nothing to fix.")
        write_outputs(out)
        return 0
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    pr = state.get("pr")
    head_sha = state.get("head_sha")
    if not pr or not head_sha or not token or not repo:
        log("fix needs a pull request, its head sha, a github-token and a repository; skipping.")
        write_outputs(out)
        return 0

    blocked = [e for e in state.get("entries") or [] if e.get("blockers")]
    if not blocked:
        log("no blockers to fix.")
        write_outputs(out)
        return 0
    if len(blocked) > 1:
        log("%d models are blocked; fixing the first, %s." % (len(blocked), blocked[0]["target"]))
    entry = blocked[0]
    target = entry["target"]
    model = next((m for m in state.get("models") or [] if m.get("target") == target), None)
    if model is None:
        log("the blocked model is not in the config any more; skipping.")
        write_outputs(out)
        return 0
    listed = [m["file"] for m in state.get("models") or []]
    policy = (state.get("policies") or {}).get(target)

    scratch = add_fix_worktree(workspace, head_sha, tmp)
    if scratch is None:
        write_outputs(out)
        return 0

    def reset():
        git(["checkout", "--", "."], scratch)
        git(["clean", "-fd"], scratch)

    def apply_fn(patches):
        ok, err = fixmod.patch_paths_allowed(patches, listed)
        if not ok:
            return False, err
        return fixmod.apply_patches(patches, scratch)

    def verify():
        result = process_model(model, scratch, None, tmp, pr, api_key, policy=policy,
                               out_root=os.path.join(tmp, "neurarch-fix-plan"))
        snap = plan_snapshot(result)
        log("re-planned %s: legal=%s params=%s blockers=%d" % (
            target, snap["legal"], snap["params"], len(snap["blockers"])))
        return snap

    sources = [
        ("the exact patch from neurarch-mcp `suggest_fix`", exact_source(scratch, [model["file"]])),
        ("a proposal from `POST /api/v1/fix`", server_source(scratch, listed, api_key, policy, listed)),
    ]
    before = {"summary": entry.get("summary"), "params": entry.get("params"),
              "blockers": entry.get("blockers")}
    outcome = fixmod.run_fix_rounds(entry["blockers"], sources, apply_fn, verify,
                                    reference_params=entry.get("params"), reset_fn=reset)

    skipped = ["The %s." % fixmod.SKIPPED_NO_KEY[len("the "):]] if not api_key else []
    if not outcome["accepted"]:
        remove_worktree(workspace, scratch)
        line = fixmod.failure_line(outcome["reason"], skipped=None)
        log("no fix pushed: %s" % outcome["reason"])
        out.update({"fix-status": "not-fixed", "fix-line": line})
        write_outputs(out)
        post_fix_line(token, repo, pr, line)
        return 0

    branch = fixmod.branch_name(pr)
    body = fixmod.fix_pr_body(pr, target, before, outcome["after"], outcome["source"],
                              outcome["patches"], outcome["rounds"], __version__,
                              notes=outcome["notes"], skipped=skipped)
    pushed = push_fix_branch(scratch, workspace, repo, token, branch, pr, target,
                             [str(p.get("path")) for p in outcome["patches"] if p.get("path")])
    remove_worktree(workspace, scratch)
    if not pushed:
        line = fixmod.failure_line("the fix passed the gate but the branch could not be pushed "
                                   "(does the token carry `contents: write`?)")
        out.update({"fix-status": "push-failed", "fix-line": line})
        write_outputs(out)
        post_fix_line(token, repo, pr, line)
        return 0

    try:
        pr_number, pr_url, updated = open_or_update_fix_pr(token, repo, branch, pr, state, body)
    except fixmod.FixError as e:
        log("could not open the fix pull request: %s" % e)
        line = fixmod.failure_line("the fix passed the gate and `%s` was pushed, but the pull "
                                   "request could not be opened (%s)" % (branch, e))
        out.update({"fix-status": "pr-failed", "fix-branch": branch, "fix-line": line})
        write_outputs(out)
        post_fix_line(token, repo, pr, line)
        return 0

    line = fixmod.success_line(pr_number, pr_url, outcome["source"], before.get("params"),
                               (outcome["after"] or {}).get("params"), updated=updated)
    out.update({"fix-status": "updated" if updated else "opened", "fix-branch": branch,
                "fix-pr-number": str(pr_number), "fix-pr-url": pr_url, "fix-line": line})
    write_outputs(out)
    post_fix_line(token, repo, pr, line)
    log("fix pull request %s: %s" % ("updated" if updated else "opened", pr_url))
    return 0


def push_fix_branch(scratch: str, workspace: str, repo: str, token: str, branch: str,
                    pr: int, target: str, paths: List[str]) -> bool:
    """Commit exactly the files the patch named, and push the branch.

    Never `git add -A`: tracing the model in this worktree imports it, which writes
    `__pycache__/*.pyc` next to the source, and the first fix pull request this bot ever opened
    carried one. The commit is the patch's own paths and nothing else.
    """
    git(["config", "user.name", "neurarch-bot"], scratch)
    git(["config", "user.email", "neurarch-bot@users.noreply.github.com"], scratch)
    for path in sorted(set(paths)):
        git(["add", "--", path], scratch)
    message = "fix: resolve the blocker neurarch-bot found on #%d\n\nModel: %s\n" % (pr, target)
    r = git(["commit", "-m", message], scratch)
    if r.returncode != 0:
        log("nothing to commit for the fix branch: %s" % (r.stdout or r.stderr).strip())
        return False
    server = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
    remote = "%s/%s.git" % (server.rstrip("/"), repo)
    remote = remote.replace("https://", "https://x-access-token:%s@" % token, 1)
    r = git(["push", "--force", remote, "HEAD:refs/heads/%s" % branch], scratch)
    if r.returncode != 0:
        log("push failed: %s" % _scrub(r.stderr.strip(), token))
        return False
    return True


def _scrub(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def open_or_update_fix_pr(token: str, repo: str, branch: str, pr: int,
                          state: Dict[str, Any], body: str) -> Tuple[int, str, bool]:
    """Open the fix pull request, or update the one already open for this branch.

    Never a second pull request for the same original: the branch name is derived from the PR
    number, so a re-run pushes the same branch and this finds the pull request already on it.
    """
    owner = repo.split("/")[0]
    open_prs = fixmod.gh_request("GET", "/repos/%s/pulls?state=open&head=%s:%s&per_page=100"
                                 % (repo, owner, branch), token) or []
    existing = fixmod.existing_fix_pr(open_prs, branch)
    title = "Fix the blocker in #%d" % pr
    if existing:
        fixmod.gh_request("PATCH", "/repos/%s/pulls/%d" % (repo, existing["number"]), token,
                          {"title": title, "body": body})
        return int(existing["number"]), str(existing.get("html_url") or ""), True
    head_ref = state.get("head_ref") or _head_ref_from_event()
    if not head_ref:
        raise FixNoHeadRef("the pull request's head branch name is not in the event payload")
    created = fixmod.gh_request("POST", "/repos/%s/pulls" % repo, token, {
        "title": title, "body": body, "head": branch, "base": head_ref, "maintainer_can_modify": True})
    if not isinstance(created, dict) or not created.get("number"):
        raise fixmod.FixError("GitHub did not return the new pull request")
    return int(created["number"]), str(created.get("html_url") or ""), False


class FixNoHeadRef(fixmod.FixError):
    pass


def _head_ref_from_event() -> Optional[str]:
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        event = json.load(f)
    return ((event.get("pull_request") or {}).get("head") or {}).get("ref")


def post_fix_line(token: str, repo: str, pr: int, line: str) -> None:
    """Add the one fix line to the plan comment the previous step posted."""
    try:
        comments = fixmod.gh_request("GET", "/repos/%s/issues/%d/comments?per_page=100" % (repo, pr),
                                     token) or []
        target = next((c for c in comments if isinstance(c, dict)
                       and str(c.get("body") or "").startswith(plan_request.MARKER)), None)
        if target is None:
            log("no plan comment to add the fix line to.")
            return
        body = fixmod.comment_with_fix_line(str(target.get("body") or ""), line)
        fixmod.gh_request("PATCH", "/repos/%s/issues/comments/%d" % (repo, target["id"]), token,
                          {"body": body})
        log("added the fix line to comment %s" % target["id"])
    except fixmod.FixError as e:
        log("could not update the plan comment with the fix line: %s" % e)


def main() -> int:
    workspace = os.environ.get("GITHUB_WORKSPACE") or os.getcwd()
    tmp = os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    if (os.environ.get("NEURARCH_STAGE") or "").strip().lower() == "fix":
        return run_fix(workspace, tmp, (os.environ.get("INPUT_API_KEY") or "").strip() or None)
    mode = (os.environ.get("INPUT_MODE") or "pr").strip().lower()
    if mode not in ("pr", "report"):
        raise SystemExit("neurarch-bot: `mode` must be `pr` or `report`, not `%s`" % mode)
    config_path = os.path.join(workspace, os.environ.get("INPUT_CONFIG") or ".neurarch.yml")
    api_key = (os.environ.get("INPUT_API_KEY") or "").strip() or None
    fail_on_warn = env_flag("INPUT_FAIL_ON_WARN")
    comment_on_clean = env_flag("INPUT_COMMENT_ON_CLEAN")
    comment_path = os.path.join(tmp, "neurarch-bot-comment.md")

    event: Dict[str, Any] = {}
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.isfile(event_path):
        with open(event_path, "r", encoding="utf-8") as f:
            event = json.load(f)
    pr_obj = event.get("pull_request") or {}
    pr = pr_obj.get("number")
    base_sha = os.environ.get("NEURARCH_BASE_SHA") or (pr_obj.get("base") or {}).get("sha")
    head_sha = os.environ.get("NEURARCH_HEAD_SHA") or (pr_obj.get("head") or {}).get("sha")
    run_url = None
    if os.environ.get("GITHUB_SERVER_URL") and os.environ.get("GITHUB_REPOSITORY") and os.environ.get("GITHUB_RUN_ID"):
        run_url = "%s/%s/actions/runs/%s" % (os.environ["GITHUB_SERVER_URL"], os.environ["GITHUB_REPOSITORY"],
                                             os.environ["GITHUB_RUN_ID"])

    base_outputs = {"legal": "true", "blockers": "0", "warnings": "0", "plans-json": "[]",
                    "fail": "false", "comment": "none", "comment-path": comment_path, "affected": "0",
                    "report": "none", "report-body-path": "", "report-comment-path": "",
                    "report-title": report.TITLE, "state-path": ""}

    config = load_config(config_path)
    if config is None:
        log("no %s in this repository; nothing to plan." % os.path.basename(config_path))
        print("""
Add a .neurarch.yml at the repository root listing the models to plan on every pull request:

  models:
    - path: models/gpt.py:TinyGPT     # file.py:ClassOrFactory (called with no arguments)
      input: "1,128:long"             # batch first; :long for token ids
    - path: models/cnn.py:SmallCNN
      input: 1,3,32,32

Optional per model: extra_inputs (a list of more --input specs for multi-input forwards),
depth (stop descending at this module depth), name, policy (this model's own house rules).

A top-level `policy:` block applies to every model and turns the organisation's own lines
into blockers and warnings in the same plan:

  policy:
    max_params: 200M
    must_fit: A100-40GB
    forbid_types: [softmax]
""")
        write_outputs(base_outputs)
        return 0

    models = parse_models(config)
    policies = {m["target"]: parse_policy(config, m) for m in models}
    for target, pol in policies.items():
        if pol:
            log("policy for %s: %s" % (target, json.dumps(pol, sort_keys=True)))
    if mode == "report":
        return run_report(models, workspace, tmp, api_key, config_path, run_url, base_outputs,
                          policies=policies)

    if not pr or not base_sha or not head_sha:
        log("not a pull_request event (no base/head sha); planning every listed model against no base.")
        changed = [m["file"] for m in models]
    else:
        changed = changed_files(workspace, base_sha, head_sha)
        log("%d changed file(s) in PR #%s" % (len(changed), pr))
    listed_files = [m["file"] for m in models]
    affected = [m for m in models if is_affected(m, changed, listed_files)]
    log("%d of %d listed model(s) affected: %s" % (len(affected), len(models),
                                                   ", ".join(m["target"] for m in affected) or "none"))

    if not affected:
        note = "No listed model changed in this PR (%d model%s in `%s`)." % (
            len(models), "" if len(models) == 1 else "s", os.path.basename(config_path))
        with open(comment_path, "w", encoding="utf-8") as f:
            f.write(plan_request.assemble_comment([], __version__, run_url, note=note))
        out = dict(base_outputs)
        out["comment"] = "update_only"
        write_outputs(out)
        return 0

    base_dir = add_base_worktree(workspace, base_sha, tmp) if base_sha and head_sha else None
    entries: List[Dict[str, Any]] = []
    try:
        for m in affected:
            entries.append(process_model(m, workspace, base_dir, tmp, pr, api_key,
                                         policy=policies.get(m["target"])))
    finally:
        remove_worktree(workspace, base_dir)

    blockers = sum(e["blockers"] for e in entries)
    warnings = sum(e["warnings"] for e in entries)
    legal = all(e["legal"] for e in entries)
    fail = (not legal) or (fail_on_warn and warnings > 0)
    all_clean = all(is_clean(e) for e in entries)
    comment_mode = "create_or_update"
    if all_clean and not comment_on_clean:
        comment_mode = "update_only"
        log("every affected model is structurally unchanged and clean; not creating a comment "
            "(set comment-on-clean: true to post anyway).")

    body = plan_request.assemble_comment(entries, __version__, run_url)
    with open(comment_path, "w", encoding="utf-8") as f:
        f.write(body)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(body.replace(plan_request.MARKER + "\n", "", 1))

    state_path = os.path.join(tmp, "neurarch-bot-state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"pr": pr, "head_sha": head_sha, "base_sha": base_sha,
                   "config": os.environ.get("INPUT_CONFIG") or ".neurarch.yml",
                   "head_ref": (pr_obj.get("head") or {}).get("ref"),
                   "models": affected, "policies": policies,
                   "entries": [fix_state_entry(e) for e in entries]}, f)

    write_outputs({
        "legal": "true" if legal else "false",
        "blockers": str(blockers),
        "warnings": str(warnings),
        "plans-json": json.dumps([summary_row(e) for e in entries], separators=(",", ":")),
        "fail": "true" if fail else "false",
        "comment": comment_mode,
        "comment-path": comment_path,
        "affected": str(len(entries)),
        "state-path": state_path,
    })
    log("%d model(s) planned: %d blocker(s), %d warning(s), legal=%s, fail=%s" % (
        len(entries), blockers, warnings, legal, fail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
