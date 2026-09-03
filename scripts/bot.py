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
        })
    return out


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
                  pr: Optional[int], api_key: Optional[str]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"target": model["target"], "input": model["input"], "status": "planned",
                             "blockers": 0, "warnings": 0, "legal": True}
    out_dir = os.path.join(tmp, "neurarch-bot", slug(model["target"]))
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
        response = plan_request.request_plan(head_graph, base_graph, source, api_key=api_key)
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
               config_path: str, run_url: Optional[str], base_outputs: Dict[str, str]) -> int:
    """Report mode: every listed model at this commit, one issue body, no verdict on the job."""
    previous = read_previous_report(os.environ.get("NEURARCH_PREVIOUS_REPORT"))
    log("planning all %d listed model(s) at the current commit: no base, no diff against a branch."
        % len(models))
    entries = [process_model(m, workspace, None, tmp, None, api_key) for m in models]
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


def main() -> int:
    workspace = os.environ.get("GITHUB_WORKSPACE") or os.getcwd()
    tmp = os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
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
                    "report-title": report.TITLE}

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
depth (stop descending at this module depth), name.
""")
        write_outputs(base_outputs)
        return 0

    models = parse_models(config)
    if mode == "report":
        return run_report(models, workspace, tmp, api_key, config_path, run_url, base_outputs)

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
            entries.append(process_model(m, workspace, base_dir, tmp, pr, api_key))
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

    write_outputs({
        "legal": "true" if legal else "false",
        "blockers": str(blockers),
        "warnings": str(warnings),
        "plans-json": json.dumps([summary_row(e) for e in entries], separators=(",", ":")),
        "fail": "true" if fail else "false",
        "comment": comment_mode,
        "comment-path": comment_path,
        "affected": str(len(entries)),
    })
    log("%d model(s) planned: %d blocker(s), %d warning(s), legal=%s, fail=%s" % (
        len(entries), blockers, warnings, legal, fail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
