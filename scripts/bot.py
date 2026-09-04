"""neurarch-bot driver: read `.neurarch.yml`, trace the models, ask the API about each, and
write one comment (or one issue) body plus the step outputs.

Four modes over one config-parse-and-trace path. `pr` finds the models this pull request
touches, traces HEAD and BASE, and writes a comment that a blocker turns red. `report` traces
every listed model at the current commit with no base, and writes the weekly issue body, which
never fails the job. `train` answers a `/neurarch train` comment on a pull request: it traces
the one model asked for and submits a real GPU run, then edits its own comment as the epochs
arrive. `init` is the one mode that runs when there is no config at all: it walks the repository,
traces what it can, and opens a pull request adding the `.neurarch.yml` the other three need.
`pr` and `report` walk `process_model`; `train` walks `run_train` and `scripts/train.py`;
`init` walks `run_init` and `scripts/init.py`.

Runs inside the composite action after torch and neurarch-trace are installed. Everything that
leaves the machine goes through `plan_request` or `train`; this file never reads a weight.
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
import init as initmod  # noqa: E402
import plan_request  # noqa: E402
import policy_lines  # noqa: E402
import report  # noqa: E402
import train as trainmod  # noqa: E402

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


def add_worktree(workspace: str, sha: str, tmp: str, name: str) -> Optional[str]:
    """A detached checkout of one commit, replacing any left behind by an earlier run."""
    path = os.path.join(tmp, name)
    if os.path.isdir(path):
        git(["worktree", "remove", "--force", path], workspace)
        shutil.rmtree(path, ignore_errors=True)
    r = git(["worktree", "add", "--detach", path, sha], workspace)
    if r.returncode != 0:
        log("could not check out %s: %s" % (sha[:10], r.stderr.strip()))
        return None
    return path


def add_base_worktree(workspace: str, base_sha: str, tmp: str) -> Optional[str]:
    return add_worktree(workspace, base_sha, tmp, "neurarch-base")


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


def attach_rule_locations(entries: List[Dict[str, Any]], config_path: str, config_rel: str,
                          repo: Optional[str], sha: Optional[str]) -> None:
    """Tell each entry where the policy rules it broke are written.

    The verdict names the rule; this names the decision. Read from the config file this run
    actually sent, at the commit that was planned, so the line number is the one the reader
    sees when they open it from the pull request.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        log("could not re-read %s for rule locations: %s" % (config_rel, e))
        return
    locations = policy_lines.locate_rules(text)
    blob = None
    if repo and sha:
        blob = "%s/%s/blob/%s" % ((os.environ.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/"),
                                  repo, sha)
    for entry in entries:
        plan = entry.get("plan")
        if not isinstance(plan, dict):
            continue
        lines = policy_lines.rule_lines(plan, locations, entry.get("target") or "", config_rel, blob)
        if lines:
            entry["rule_lines"] = lines
            log("%s broke %d policy rule(s); the comment names the line each is written on."
                % (entry.get("target"), len(lines)))


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
    return add_worktree(workspace, head_sha, tmp, "neurarch-fix")


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


# ---------------------------------------------------------------- init stage


def measure_graph(graph: Dict[str, Any], target: str, input_spec: str,
                  api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Params, layers and GPU fit for one traced graph, from the same endpoint that will
    later enforce the rules derived from them. No policy is sent: nothing is being judged."""
    try:
        response = plan_request.request_plan(
            graph, None, plan_request.build_source(target, input_spec, None), api_key=api_key)
    except plan_request.PlanRequestError as e:
        log("could not plan %s: %s" % (target, e))
        return None
    plan = response.get("plan") or {}
    model = plan.get("model") or {}
    cost = plan.get("cost") or {}
    fits = initmod.smallest_gpu([{"gpu_fits": cost.get("gpuFits") or {}}])
    return {"target": target, "params": model.get("params"), "layers": model.get("layers"),
            "gpu_fits": cost.get("gpuFits") or {}, "fits": fits[0] if fits else None}


def init_candidate(candidate: Dict[str, Any], workspace: str, tmp: str,
                   api_key: Optional[str]) -> Dict[str, Any]:
    """Trace one discovered class with each guessed input in turn, then measure what traced."""
    entry = {"target": candidate["target"], "file": candidate["file"], "name": candidate["name"],
             "input": candidate["inputs"][0], "status": "skipped"}
    if candidate.get("part_of"):
        entry["reason"] = ("`%s` constructs it in the same file, so it is a part of that model "
                           "rather than a model. Uncomment it if you plan it on its own."
                           % candidate["part_of"])
        return entry
    if candidate["needs_args"]:
        entry["reason"] = ("`%s()` takes %s, so it cannot be constructed with no arguments. "
                           "Add a zero-argument factory beside it and list that instead."
                           % (candidate["name"], ", ".join("`%s`" % a for a in candidate["needs_args"])))
        return entry
    out_dir = os.path.join(tmp, "neurarch-init", slug(candidate["target"]))
    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(out_dir, "head.neurarch.json")
    last = ""
    for spec in candidate["inputs"]:
        model = {"target": candidate["target"], "file": candidate["file"], "input": spec,
                 "extra_inputs": [], "depth": None, "name": None}
        log("tracing %s (guessed input %s)" % (candidate["target"], spec))
        ok, msg = trace(model, workspace, graph_path)
        if ok:
            entry["input"] = spec
            with open(graph_path, "r", encoding="utf-8") as f:
                graph = json.load(f)
            measured = measure_graph(graph, candidate["target"], spec, api_key)
            if measured is None:
                entry["status"] = "traced"
                return entry
            entry.update({"status": "measured", "params": measured["params"],
                          "layers": measured["layers"], "fits": measured["fits"],
                          "gpu_fits": measured["gpu_fits"]})
            return entry
        last = (msg or "").splitlines()[-1] if msg else "no output"
    entry["reason"] = ("it did not trace with %s (the input spec is a guess read off its first "
                       "layer). Last error: `%s`."
                       % (" or ".join("`%s`" % s for s in candidate["inputs"]), last[:200]))
    return entry


def init_graph_entry(rel: str, workspace: str, api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """A `*.neurarch.json` already in the repository, measured but never listed: a config
    entry is a python entry point, and a graph file has none."""
    path = os.path.join(workspace, rel)
    try:
        with open(path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except (OSError, ValueError) as e:
        log("could not read %s: %s" % (rel, e))
        return None
    return measure_graph(graph, rel, "", api_key)


def run_init(workspace: str, tmp: str, api_key: Optional[str], config_path: str,
             base_outputs: Dict[str, str], run_url: Optional[str]) -> int:
    """`mode: init`: write the config this repository does not have, as a pull request.

    Does nothing at all when a config already exists, and never writes to a branch anybody
    works on. Never returns non-zero: a repository with nothing to find is not a broken run.
    """
    name = os.path.basename(config_path)
    out = dict(base_outputs)
    out.update({"init-status": "none", "init-pr-url": "", "init-pr-number": ""})
    if os.path.isfile(config_path):
        log("%s already exists; init does nothing. Delete it first if you want it rewritten." % name)
        out["init-status"] = "exists"
        write_outputs(out)
        return 0

    found = initmod.discover(workspace)
    candidates = found["candidates"]
    graphs = found["graphs"]
    log("walked %d python file(s): %d candidate model(s), %d traced graph(s)."
        % (found["python_files"], len(candidates), len(graphs)))
    if not candidates and not graphs:
        log("nothing that looks like a model in this repository; writing nothing.")
        out["init-status"] = "nothing-found"
        write_outputs(out)
        return 0

    # The trace budget is spent on things that look like models: a `Block` another class in the
    # same file constructs is answered without tracing it, so a transformer file cannot eat the
    # budget with its own parts.
    whole = [c for c in candidates if not c.get("part_of")]
    parts = [c for c in candidates if c.get("part_of")]
    entries = [init_candidate(c, workspace, tmp, api_key) for c in whole[:initmod.MAX_TRACE]]
    for extra in whole[initmod.MAX_TRACE:]:
        entries.append({"target": extra["target"], "file": extra["file"], "name": extra["name"],
                        "input": extra["inputs"][0], "status": "skipped",
                        "reason": "found, but this run traces the first %d models it sees; "
                                  "uncomment it to have it planned too." % initmod.MAX_TRACE})
    entries.extend(init_candidate(c, workspace, tmp, api_key) for c in parts)
    measured = [e for e in entries if e.get("status") == "measured"]
    for rel in graphs[:initmod.MAX_TRACE]:
        row = init_graph_entry(rel, workspace, api_key)
        if row:
            measured.append(row)

    derived = initmod.derive_policy(measured)
    text = initmod.render_config(entries, derived, __version__, graphs=graphs)
    body = initmod.pr_body(entries, derived, __version__, config_name=name, graphs=graphs,
                           run_url=run_url, api_host=plan_request.api_host(plan_request.api_base()),
                           searched=found["python_files"])
    log("derived %s from %d measured model(s)."
        % (", ".join("%s: %s" % (k, v["value"]) for k, v in sorted(derived.items())), len(measured)))
    print(text, flush=True)

    config_out = os.path.join(tmp, "neurarch-init-config.yml")
    with open(config_out, "w", encoding="utf-8") as f:
        f.write(text)
    body_path = os.path.join(tmp, "neurarch-init-body.md")
    with open(body_path, "w", encoding="utf-8") as f:
        f.write(body)
    out["init-config-path"] = config_out
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(body.replace(initmod.MARKER + "\n", "", 1))

    token = (os.environ.get("INPUT_GITHUB_TOKEN") or "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    if not token or not repo:
        log("no github-token or repository, so the config was written to %s and nowhere else."
            % config_out)
        out["init-status"] = "no-token"
        write_outputs(out)
        return 0
    status, number, url = open_init_pr(workspace, tmp, name, text, body, token, repo)
    out.update({"init-status": status, "init-pr-number": str(number or ""), "init-pr-url": url or ""})
    write_outputs(out)
    return 0


def default_branch(repo: str, token: str) -> Optional[str]:
    try:
        data = fixmod.gh_request("GET", "/repos/%s" % repo, token)
    except fixmod.FixError as e:
        log("could not read the default branch (%s)." % e)
        return None
    return str(data.get("default_branch")) if isinstance(data, dict) and data.get("default_branch") else None


def current_branch() -> Optional[str]:
    ref = os.environ.get("GITHUB_REF") or ""
    if ref.startswith("refs/heads/"):
        return ref[len("refs/heads/"):]
    return None


def open_init_pr(workspace: str, tmp: str, config_name: str, text: str, body: str,
                 token: str, repo: str) -> Tuple[str, Optional[int], Optional[str]]:
    """Commit the config on a branch of its own and open the pull request.

    The base is the branch this ran on, because that is the tree that was measured; the branch
    written to is never it. A default branch is where a repository's own work lives and a bot
    with a write token does not get to put a file there because it thinks the file is a good
    idea.
    """
    default = default_branch(repo, token)
    base = current_branch() or default
    if not base:
        log("could not work out which branch to open the pull request into; writing nothing.")
        return "no-base", None, None
    branch = initmod.branch_name(base, default)
    if branch == base:
        log("the branch to push (%s) is the branch to open into; refusing to commit to it." % branch)
        return "refused-same-branch", None, None

    head_sha = os.environ.get("GITHUB_SHA") or "HEAD"
    scratch = add_worktree(workspace, head_sha, tmp, "neurarch-init")
    if scratch is None:
        return "worktree-failed", None, None
    try:
        with open(os.path.join(scratch, config_name), "w", encoding="utf-8") as f:
            f.write(text)
        git(["config", "user.name", "neurarch-bot"], scratch)
        git(["config", "user.email", "neurarch-bot@users.noreply.github.com"], scratch)
        git(["add", "--", config_name], scratch)
        message = "Add %s: the models in this repository, and the house rules for them\n" % config_name
        r = git(["commit", "-m", message], scratch)
        if r.returncode != 0:
            log("nothing to commit: %s" % (r.stdout or r.stderr).strip())
            return "nothing-to-commit", None, None
        server = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
        remote = ("%s/%s.git" % (server.rstrip("/"), repo)).replace(
            "https://", "https://x-access-token:%s@" % token, 1)
        r = git(["push", "--force", remote, "HEAD:refs/heads/%s" % branch], scratch)
        if r.returncode != 0:
            log("push failed: %s" % _scrub(r.stderr.strip(), token))
            return "push-failed", None, None
    finally:
        remove_worktree(workspace, scratch)

    owner = repo.split("/")[0]
    try:
        open_prs = fixmod.gh_request("GET", "/repos/%s/pulls?state=open&head=%s:%s&per_page=100"
                                     % (repo, owner, branch), token) or []
        existing = fixmod.existing_fix_pr(open_prs, branch)
        if existing:
            fixmod.gh_request("PATCH", "/repos/%s/pulls/%d" % (repo, existing["number"]), token,
                              {"title": initmod.TITLE, "body": body})
            log("updated the config pull request: %s" % existing.get("html_url"))
            return "updated", int(existing["number"]), str(existing.get("html_url") or "")
        created = fixmod.gh_request("POST", "/repos/%s/pulls" % repo, token, {
            "title": initmod.TITLE, "body": body, "head": branch, "base": base,
            "maintainer_can_modify": True})
    except fixmod.FixError as e:
        log("the branch %s was pushed but the pull request could not be opened (%s). "
            "Is \"Allow GitHub Actions to create and approve pull requests\" on?" % (branch, e))
        return "pr-failed", None, None
    if not isinstance(created, dict) or not created.get("number"):
        return "pr-failed", None, None
    log("opened the config pull request: %s" % created.get("html_url"))
    return "opened", int(created["number"]), str(created.get("html_url") or "")


# ---------------------------------------------------------------- train stage


def read_event() -> Dict[str, Any]:
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def post_comment(token: str, repo: str, issue: int, body: str) -> Optional[int]:
    try:
        created = fixmod.gh_request("POST", "/repos/%s/issues/%d/comments" % (repo, issue),
                                    token, {"body": body})
    except fixmod.FixError as e:
        log("could not post the comment: %s" % e)
        return None
    return int(created["id"]) if isinstance(created, dict) and created.get("id") else None


def patch_comment(token: str, repo: str, comment_id: int, body: str) -> bool:
    try:
        fixmod.gh_request("PATCH", "/repos/%s/issues/comments/%d" % (repo, comment_id), token,
                          {"body": body})
    except fixmod.FixError as e:
        log("could not update comment %s: %s" % (comment_id, e))
        return False
    return True


def upsert_train_comment(token: str, repo: str, issue: int, marker: str,
                         body: str) -> Optional[int]:
    """Find this run's own comment by its job marker, or create it.

    A re-run of the same job (a step that was retried, a workflow re-dispatched) edits the
    comment already on the pull request rather than posting a second one.
    """
    try:
        comments = fixmod.gh_request(
            "GET", "/repos/%s/issues/%d/comments?per_page=100" % (repo, issue), token) or []
    except fixmod.FixError as e:
        log("could not list comments: %s" % e)
        comments = []
    existing = trainmod.find_comment(comments, marker)
    if existing:
        return int(existing["id"]) if patch_comment(token, repo, int(existing["id"]), body) else None
    return post_comment(token, repo, issue, body)


def run_train(workspace: str, tmp: str, api_key: Optional[str]) -> int:
    """`/neurarch train` on a pull request comment: one real GPU run, or one reason there is not.

    Never returns non-zero. A refusal is an answer and a failed training run is a fact about
    the model, not a reason to turn someone's pull request check red.
    """
    host = plan_request.api_host(plan_request.api_base())
    version = __version__
    out = {"legal": "true", "blockers": "0", "warnings": "0", "plans-json": "[]", "fail": "false",
           "comment": "none", "affected": "0", "report": "none", "job-id": "", "train-status": "none"}
    event = read_event()
    comment = event.get("comment") or {}
    issue = event.get("issue") or {}
    body = comment.get("body")
    pr = issue.get("number")
    token = (os.environ.get("INPUT_GITHUB_TOKEN") or "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    run_url = None
    if os.environ.get("GITHUB_SERVER_URL") and repo and os.environ.get("GITHUB_RUN_ID"):
        run_url = "%s/%s/actions/runs/%s" % (os.environ["GITHUB_SERVER_URL"], repo,
                                             os.environ["GITHUB_RUN_ID"])

    if not trainmod.is_command(body):
        log("this comment is not `%s`; nothing to do." % trainmod.COMMAND)
        write_outputs(out)
        return 0
    if not issue.get("pull_request"):
        log("`%s` only runs on a pull request comment; this is an issue." % trainmod.COMMAND)
        write_outputs(out)
        return 0
    if not pr or not token or not repo:
        log("train needs a pull request, a github-token and a repository; skipping.")
        write_outputs(out)
        return 0

    def refuse(reason: str, status: str) -> int:
        post_comment(token, repo, int(pr), trainmod.refusal_comment(reason, version, host, run_url))
        log("refused: %s" % status)
        out["train-status"] = status
        write_outputs(out)
        return 0

    actor = (comment.get("user") or {}).get("login")
    association = comment.get("author_association")
    if not trainmod.may_spend(association):
        # First, before anything is traced or asked: the key is not the commenter's to spend.
        return refuse(trainmod.access_refusal(actor, association), "refused-access")
    if not api_key:
        return refuse(trainmod.no_key_refusal(), "refused-no-key")

    args, arg_error = trainmod.parse_command(str(body))
    if arg_error:
        return refuse("**I could not read that command.** %s Nothing was submitted." % arg_error,
                      "refused-arguments")

    config_path = os.path.join(workspace, os.environ.get("INPUT_CONFIG") or ".neurarch.yml")
    config = load_config(config_path)
    if config is None:
        return refuse("**No `%s` in this repository**, so there is nothing to train. Nothing was "
                      "submitted." % os.path.basename(config_path), "refused-no-config")
    models = parse_models(config)
    model, choose_error = trainmod.choose_model(models, args.get("model"))
    if choose_error:
        return refuse("**%s** Nothing was submitted." % choose_error, "refused-model")

    dataset = (args.get("dataset") or "mnist").lower()
    train_config = trainmod.config_from_args(args)
    gpu = args.get("gpu")
    target = model["target"]
    log("%s asked for %s on %s (%s)" % (actor, target, dataset,
                                        json.dumps(train_config, sort_keys=True) or "defaults"))

    out_dir = os.path.join(tmp, "neurarch-train", slug(target))
    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(out_dir, "head.neurarch.json")
    ok, msg = trace(model, workspace, graph_path)
    if not ok:
        return refuse("**Could not trace `%s` with input `%s`, so there is no graph to train.** "
                      "Nothing was submitted.\n\n```\n%s\n```"
                      % (target, model["input"], trainmod.trim_error(msg)), "refused-trace")
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    # The house rules first. `POST /api/v1/train` plans the graph on its own and never sees a
    # policy, so a `max_params` line in `.neurarch.yml` would be discovered only after the GPU
    # was booked. `/api/v1/plan` costs nothing and answers the same question.
    policy = parse_policy(config, model)
    if policy:
        log("checking this repository's own policy before spending anything: %s"
            % json.dumps(policy, sort_keys=True))
        try:
            planned = plan_request.request_plan(
                graph, None, plan_request.build_source(target, model["input"], pr),
                api_key=api_key, policy=policy)
        except plan_request.PlanRequestError as e:
            return refuse("**The policy check could not be run (%s), so nothing was submitted.** "
                          "`.neurarch.yml` sets house rules for this model and the bot will not "
                          "spend the key owner's GPU budget without checking them." % e,
                          "refused-plan-failed")
        if plan_request.has_blocker(planned["plan"]):
            post_comment(token, repo, int(pr), trainmod.blocked_comment(
                target, planned["plan"], planned.get("text", ""), version, host,
                source="policy", run_url=run_url))
            log("refused by this repository's own policy; nothing submitted.")
            out["train-status"] = "refused-policy"
            write_outputs(out)
            return 0

    source = trainmod.build_source(target, repo, pr, version)
    try:
        status, data, message = trainmod.submit_train(
            graph, dataset, plan_request.api_base(), api_key, config=train_config, gpu=gpu,
            source=source)
    except trainmod.TrainRequestError as e:
        return refuse("**`%s` could not reach the training endpoint: %s.** Nothing was submitted."
                      % (trainmod.COMMAND, e), "refused-unreachable")

    if status == 422 and data.get("error") == "blocked":
        post_comment(token, repo, int(pr), trainmod.blocked_comment(
            target, data.get("plan") or {}, data.get("text", ""), version, host,
            source="server", run_url=run_url))
        log("the endpoint refused the design; nothing was submitted and nothing was spent.")
        out["train-status"] = "refused-blocked"
        write_outputs(out)
        return 0
    if status == 503:
        post_comment(token, repo, int(pr), trainmod.gpu_unconfigured_comment(
            data.get("message") or message, version, host, run_url))
        out["train-status"] = "refused-no-gpu"
        write_outputs(out)
        return 0
    if status != 200 or not data.get("job_id"):
        post_comment(token, repo, int(pr),
                     trainmod.error_comment(status, message, version, host, run_url))
        out["train-status"] = "refused-error"
        write_outputs(out)
        return 0

    job_id = str(data["job_id"])
    clamped = data.get("config") if isinstance(data.get("config"), dict) else {}
    clamp_note = trainmod.clamp_note(train_config, clamped)
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else None
    marker = trainmod.JOB_MARKER % job_id
    log("submitted %s: job %s (%s)" % (target, job_id, json.dumps(clamped, sort_keys=True)))
    out["job-id"] = job_id

    def render(job: Optional[Dict[str, Any]], timed_out: bool = False) -> str:
        return trainmod.run_comment(job_id, target, dataset, clamped or train_config, plan,
                                    version, host, job=job, gpu=gpu, clamped=clamp_note,
                                    timed_out=timed_out, run_url=run_url)

    comment_id = upsert_train_comment(token, repo, int(pr), marker, render(None))

    def on_change(job: Dict[str, Any]) -> None:
        log("job %s: %s, %d epoch row(s)" % (job_id, job.get("status"), len(trainmod.epoch_rows(job))))
        if comment_id:
            patch_comment(token, repo, comment_id, render(job))

    timeout_min = _timeout_min()
    job, timed_out = trainmod.follow_job(job_id, plan_request.api_base(), api_key, on_change,
                                         timeout_min=timeout_min)
    if comment_id:
        patch_comment(token, repo, comment_id, render(job, timed_out=timed_out))
    status_name = "timeout" if timed_out else str((job or {}).get("status") or "unknown")
    out["train-status"] = status_name
    write_outputs(out)
    log("job %s ended: %s" % (job_id, status_name))
    return 0


def _timeout_min() -> float:
    raw = (os.environ.get("INPUT_TIMEOUT_MIN") or "").strip()
    try:
        value = float(raw)
    except ValueError:
        return float(trainmod.DEFAULT_TIMEOUT_MIN)
    return value if value > 0 else float(trainmod.DEFAULT_TIMEOUT_MIN)


def main() -> int:
    workspace = os.environ.get("GITHUB_WORKSPACE") or os.getcwd()
    tmp = os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    if (os.environ.get("NEURARCH_STAGE") or "").strip().lower() == "fix":
        return run_fix(workspace, tmp, (os.environ.get("INPUT_API_KEY") or "").strip() or None)
    mode = (os.environ.get("INPUT_MODE") or "pr").strip().lower()
    # A comment event is never a plan run: the plan has no base and no diff to compute, and
    # `mode: train` is what the caller meant. Forgiving on purpose, and it cannot misfire,
    # because train mode does nothing at all unless the comment is the command.
    if mode == "pr" and (os.environ.get("GITHUB_EVENT_NAME") or "") == "issue_comment":
        log("issue_comment event: running in train mode.")
        mode = "train"
    if mode == "train":
        return run_train(workspace, tmp, (os.environ.get("INPUT_API_KEY") or "").strip() or None)
    if mode not in ("pr", "report", "init"):
        raise SystemExit("neurarch-bot: `mode` must be `pr`, `report`, `train` or `init`, not `%s`" % mode)
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

    if mode == "init":
        return run_init(workspace, tmp, api_key, config_path, base_outputs, run_url)

    config = load_config(config_path)
    if config is None:
        log("no %s in this repository; nothing to plan." % os.path.basename(config_path))
        print("""
Run this Action once with `mode: init` and it writes the config for you: it walks the
repository, traces every model it can construct, and opens a pull request adding a
.neurarch.yml with those models and house rules derived from what it measured.

Or write it by hand. It lists the models to plan on every pull request:

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

Every supported key, annotated, with what blocks and what warns:
https://github.com/neurarch-ai/neurarch-bot/blob/main/examples/.neurarch.yml
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
    attach_rule_locations(entries, config_path, os.environ.get("INPUT_CONFIG") or ".neurarch.yml",
                          os.environ.get("GITHUB_REPOSITORY"), head_sha)

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
