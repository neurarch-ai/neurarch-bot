"""Unit tests for fix.py: patch application and refusal, the 10 percent parameter guard, the
round limit, the "already has an open fix PR" path, and the two comment lines.

No torch, no GitHub, no network beyond a stub HTTP server on loopback, and no neurarch-mcp:
every source is an injected callable, which is the reason `run_fix_rounds` takes them that way.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix  # noqa: E402

ORIGINAL = "line one\nline two\nline three\n"
PATCH_TWO = ("--- a/small_cnn.py\n"
             "+++ b/small_cnn.py\n"
             "@@ -1,3 +1,3 @@\n"
             " line one\n"
             "-line two\n"
             "+line TWO\n"
             " line three\n")


def blocker(title="policy: max_params 500K exceeded"):
    return {"title": title, "stage": "policy", "detail": "max_params: 500K"}


def plan_result(legal=True, params=141354, blockers=None, status="planned", summary="SmallCNN: will run"):
    return {"status": status, "legal": legal, "params": params, "summary": summary,
            "blockers": blockers or [], "findings": blockers or []}


# ---------------------------------------------------------------- patches

class NormalizeDiffTests(unittest.TestCase):
    def test_headers_are_rewritten_to_the_path_we_vouched_for(self):
        out = fix.normalize_diff(PATCH_TWO, "models/small_cnn.py")
        self.assertTrue(out.startswith("--- models/small_cnn.py\n+++ models/small_cnn.py\n"))
        self.assertIn("-line two", out)
        self.assertTrue(out.endswith("\n"))

    def test_only_the_first_two_headers_move(self):
        diff = PATCH_TWO + "--- not a header, part of the body\n"
        out = fix.normalize_diff(diff, "models/x.py")
        self.assertEqual(out.count("--- models/x.py"), 1)
        self.assertIn("--- not a header, part of the body", out)


class ApplyPatchTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "models"))
        with open(os.path.join(self.dir, "models", "small_cnn.py"), "w") as f:
            f.write(ORIGINAL)
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
            subprocess.run(cmd, cwd=self.dir, capture_output=True)

    def read(self):
        with open(os.path.join(self.dir, "models", "small_cnn.py")) as f:
            return f.read()

    def test_a_patch_applies(self):
        ok, err = fix.apply_patches([{"path": "models/small_cnn.py", "diff": PATCH_TWO}], self.dir)
        self.assertTrue(ok, err)
        self.assertEqual(self.read(), "line one\nline TWO\nline three\n")

    def test_a_patch_that_does_not_match_is_rejected_and_says_so(self):
        stale = PATCH_TWO.replace("line two", "line nine")
        ok, err = fix.apply_patches([{"path": "models/small_cnn.py", "diff": stale}], self.dir)
        self.assertFalse(ok)
        self.assertIn("did not apply", err)
        self.assertEqual(self.read(), ORIGINAL)

    def test_an_empty_patch_is_rejected(self):
        ok, err = fix.apply_patches([{"path": "models/small_cnn.py", "diff": "   "}], self.dir)
        self.assertFalse(ok)
        self.assertIn("empty", err)


class AllowlistTests(unittest.TestCase):
    LISTED = ["models/small_cnn.py", "models/tiny_gpt.py"]

    def test_a_listed_file_is_allowed(self):
        ok, err = fix.patch_paths_allowed([{"path": "models/small_cnn.py"}], self.LISTED)
        self.assertTrue(ok, err)

    def test_an_unlisted_file_is_refused_by_name(self):
        ok, err = fix.patch_paths_allowed([{"path": ".github/workflows/ci.yml"}], self.LISTED)
        self.assertFalse(ok)
        self.assertIn(".github/workflows/ci.yml", err)
        self.assertIn("does not list", err)

    def test_escape_attempts_are_refused(self):
        for path in ("../../etc/passwd", "/etc/passwd", "models/../../x.py", ""):
            ok, err = fix.patch_paths_allowed([{"path": path}], self.LISTED)
            self.assertFalse(ok, path)


# ---------------------------------------------------------------- the guard

class ParamGuardTests(unittest.TestCase):
    def test_within_ten_percent_passes(self):
        self.assertTrue(fix.within_tolerance(141354, 141354))
        self.assertTrue(fix.within_tolerance(150000, 141354))     # +6.1%
        self.assertTrue(fix.within_tolerance(141354 * 1.10, 141354))

    def test_past_ten_percent_fails_in_both_directions(self):
        self.assertFalse(fix.within_tolerance(141354 * 1.11, 141354))
        self.assertFalse(fix.within_tolerance(141354 * 0.5, 141354))
        self.assertFalse(fix.within_tolerance(0, 141354))

    def test_an_unknown_count_is_not_within_tolerance(self):
        self.assertFalse(fix.within_tolerance(None, 141354))
        self.assertFalse(fix.within_tolerance(141354, None))
        self.assertFalse(fix.within_tolerance(141354, 0))

    def test_judge_accepts_a_legal_plan_that_kept_its_size(self):
        ok, reason = fix.judge(plan_result(legal=True, params=141354), 141354)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "")

    def test_judge_rejects_a_legal_plan_that_gutted_the_model(self):
        ok, reason = fix.judge(plan_result(legal=True, params=40000), 141354)
        self.assertFalse(ok)
        self.assertIn("-71.7%", reason)
        self.assertIn("141.4K -> 40.0K", reason)
        self.assertIn("10% guard", reason)

    def test_judge_rejects_a_plan_that_is_still_blocked(self):
        ok, reason = fix.judge(plan_result(legal=False, blockers=[blocker()]), 141354)
        self.assertFalse(ok)
        self.assertIn("still blocked", reason)
        self.assertIn("max_params 500K", reason)

    def test_judge_rejects_a_patched_model_that_stopped_tracing(self):
        ok, reason = fix.judge(plan_result(status="trace_failed", legal=False, params=None), 141354)
        self.assertFalse(ok)
        self.assertIn("no longer traces", reason)

    def test_judge_rejects_when_it_cannot_measure(self):
        ok, reason = fix.judge(plan_result(legal=True, params=None), 141354)
        self.assertFalse(ok)
        self.assertIn("could not be compared", reason)


# ---------------------------------------------------------------- the rounds

class Recorder:
    """A stand-in worktree: records what was applied, answers with scripted plans."""

    def __init__(self, results):
        self.results = list(results)
        self.applied = []
        self.resets = 0
        self.apply_ok = True

    def apply(self, patches):
        if not self.apply_ok:
            return False, "the patch for `models/x.py` did not apply: context mismatch"
        self.applied.append([p["diff"] for p in patches])
        return True, ""

    def reset(self):
        self.resets += 1

    def verify(self):
        return self.results.pop(0) if self.results else plan_result(legal=False, blockers=[blocker()])


def source_of(name, patches, note=""):
    calls = []

    def fn(findings):
        calls.append(list(findings))
        return list(patches), note
    fn.calls = calls
    return (name, fn)


P1 = [{"path": "models/x.py", "diff": "d1", "findingTitles": ["drop the softmax"]}]
P2 = [{"path": "models/x.py", "diff": "d2", "findingTitles": ["narrow the conv"]}]


class RoundTests(unittest.TestCase):
    def test_a_legal_result_is_accepted_on_the_first_round(self):
        rec = Recorder([plan_result(legal=True, params=141354)])
        out = fix.run_fix_rounds([blocker()], [source_of("exact", P1)], rec.apply, rec.verify,
                                 reference_params=141354, reset_fn=rec.reset)
        self.assertTrue(out["accepted"])
        self.assertEqual(out["rounds"], 1)
        self.assertEqual(out["source"], "exact")
        self.assertEqual(out["patches"], P1)
        self.assertEqual(rec.applied, [["d1"]])

    def test_a_source_with_nothing_to_say_is_reported_not_retried(self):
        rec = Recorder([])
        out = fix.run_fix_rounds([blocker()], [source_of("exact", [], "no mechanical fix")],
                                 rec.apply, rec.verify, reference_params=141354, reset_fn=rec.reset)
        self.assertFalse(out["accepted"])
        self.assertEqual(rec.applied, [])
        self.assertIn("exact: no mechanical fix", out["reason"])

    def test_the_same_patch_set_is_never_tried_twice(self):
        """Three rounds of a source that keeps proposing the same diff is one attempt."""
        rec = Recorder([plan_result(legal=False, blockers=[blocker()]),
                        plan_result(legal=False, blockers=[blocker()]),
                        plan_result(legal=False, blockers=[blocker()])])
        exact = source_of("exact", P1)
        out = fix.run_fix_rounds([blocker()], [exact], rec.apply, rec.verify,
                                 reference_params=141354, reset_fn=rec.reset)
        self.assertFalse(out["accepted"])
        self.assertEqual(len(rec.applied), 1)
        self.assertEqual(out["rounds"], 1)

    def test_the_round_limit_holds_at_three(self):
        """A source that proposes something new every time still stops after MAX_ROUNDS."""
        seq = iter(range(100))
        calls = []

        def endless(findings):
            calls.append(findings)
            return [{"path": "models/x.py", "diff": "d%d" % next(seq), "findingTitles": []}], ""

        rec = Recorder([plan_result(legal=False, blockers=[blocker()]) for _ in range(10)])
        out = fix.run_fix_rounds([blocker()], [("exact", endless)], rec.apply, rec.verify,
                                 reference_params=141354, reset_fn=rec.reset)
        self.assertFalse(out["accepted"])
        self.assertEqual(len(rec.applied), fix.MAX_ROUNDS)
        self.assertEqual(len(calls), fix.MAX_ROUNDS)

    def test_the_second_source_is_tried_when_the_first_had_nothing(self):
        rec = Recorder([plan_result(legal=True, params=141354)])
        out = fix.run_fix_rounds([blocker()],
                                 [source_of("exact", [], "nothing mechanical"),
                                  source_of("server", P2)],
                                 rec.apply, rec.verify, reference_params=141354, reset_fn=rec.reset)
        self.assertTrue(out["accepted"])
        self.assertEqual(out["source"], "server")
        self.assertEqual(rec.applied, [["d2"]])

    def test_the_remaining_findings_are_fed_back_in(self):
        """Round two is asked about what round one left, not about the original blockers."""
        left = blocker("Shape problem: linear")
        rec = Recorder([plan_result(legal=False, params=141000, blockers=[left]),
                        plan_result(legal=True, params=141000)])
        second = {"path": "models/x.py", "diff": "d2", "findingTitles": []}
        answers = [(P1, ""), ([second], "")]
        seen = []

        def fn(findings):
            seen.append(list(findings))
            return answers[min(len(seen) - 1, 1)]

        out = fix.run_fix_rounds([blocker(), left], [("server", fn)], rec.apply, rec.verify,
                                 reference_params=141354, reset_fn=rec.reset)
        self.assertTrue(out["accepted"])
        self.assertEqual(out["rounds"], 2)
        self.assertEqual(seen[0], [blocker(), left])
        self.assertEqual(seen[1], [left])
        # Round one reduced the blockers from two to one, so its patch was kept and round two
        # was applied on top of it.
        self.assertEqual(rec.applied, [["d1"], ["d1", "d2"]])

    def test_a_patch_that_will_not_apply_is_a_note_not_a_crash(self):
        rec = Recorder([])
        rec.apply_ok = False
        out = fix.run_fix_rounds([blocker()], [source_of("exact", P1)], rec.apply, rec.verify,
                                 reference_params=141354, reset_fn=rec.reset)
        self.assertFalse(out["accepted"])
        self.assertIn("did not apply", out["reason"])

    def test_a_source_that_raises_is_a_note_not_a_crash(self):
        def broken(findings):
            raise fix.FixError("could not start neurarch-mcp (no node on the runner)")

        rec = Recorder([])
        out = fix.run_fix_rounds([blocker()], [("exact", broken)], rec.apply, rec.verify,
                                 reference_params=141354, reset_fn=rec.reset)
        self.assertFalse(out["accepted"])
        self.assertIn("no node on the runner", out["reason"])

    def test_the_worktree_is_reset_before_every_attempt_and_after_a_failure(self):
        rec = Recorder([plan_result(legal=False, blockers=[blocker()])])
        fix.run_fix_rounds([blocker()], [source_of("exact", P1)], rec.apply, rec.verify,
                           reference_params=141354, reset_fn=rec.reset)
        self.assertGreaterEqual(rec.resets, 2)

    def test_a_gutted_model_is_refused_even_though_the_plan_is_legal(self):
        rec = Recorder([plan_result(legal=True, params=12000)])
        out = fix.run_fix_rounds([blocker()], [source_of("exact", P1)], rec.apply, rec.verify,
                                 reference_params=141354, reset_fn=rec.reset)
        self.assertFalse(out["accepted"])
        self.assertIn("past the 10% guard", out["reason"])


# ---------------------------------------------------------------- suggest_fix parsing

class McpParsingTests(unittest.TestCase):
    def test_the_structured_result_is_picked_out_of_the_stream(self):
        stream = "\n".join([
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "neurarch-mcp"}}}),
            "not json at all",
            json.dumps({"jsonrpc": "2.0", "id": 2, "result": {
                "structuredContent": {"path": "small_cnn.py", "fixes": [
                    {"rule": "output-activation", "confidence": "proposal", "diff": PATCH_TWO,
                     "summary": "Drop the softmax call at the output."}], "notFixable": []}}}),
        ])
        result = fix._mcp_result(stream, 2)
        self.assertEqual(result["path"], "small_cnn.py")
        patches, notes = fix.exact_patches(result, "models/small_cnn.py")
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0]["path"], "models/small_cnn.py")
        self.assertTrue(patches[0]["diff"].startswith("--- models/small_cnn.py"))
        self.assertEqual(patches[0]["findingTitles"], ["Drop the softmax call at the output."])
        self.assertEqual(notes, [])

    def test_the_text_envelope_is_read_when_there_is_no_structured_content(self):
        payload = {"fixes": [], "notFixable": [{"rule": "compute-error", "reason": "a shape finding"}]}
        stream = json.dumps({"jsonrpc": "2.0", "id": 2,
                             "result": {"content": [{"type": "text", "text": json.dumps(payload)}]}})
        result = fix._mcp_result(stream, 2)
        patches, notes = fix.exact_patches(result, "models/x.py")
        self.assertEqual(patches, [])
        self.assertEqual(notes, ["compute-error: a shape finding"])

    def test_no_matching_id_is_no_result(self):
        self.assertIsNone(fix._mcp_result(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}), 2))
        self.assertIsNone(fix._mcp_result("", 2))


# ---------------------------------------------------------------- POST /api/v1/fix

class StubFix(BaseHTTPRequestHandler):
    state = {"requests": [], "mode": "ok"}

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.state["requests"].append({"path": self.path, "headers": dict(self.headers),
                                       "body": json.loads(raw.decode("utf-8"))})
        if self.state["mode"] == "401":
            payload, code = json.dumps({"error": "Fix proposals need an API key"}).encode(), 401
        elif self.state["mode"] == "nopatches":
            payload, code = json.dumps({"rationale": "nothing to do"}).encode(), 200
        else:
            payload = json.dumps({"patches": [{"path": "models/small_cnn.py", "diff": PATCH_TWO,
                                               "findingTitles": ["policy: max_params 500K exceeded"]}],
                                  "rationale": "narrowed the last block", "model": "deepseek/x"}).encode()
            code = 200
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class RequestFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), StubFix)
        cls.api = "http://127.0.0.1:%d" % cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        StubFix.state["requests"] = []
        StubFix.state["mode"] = "ok"

    def test_body_and_key_header(self):
        data = fix.request_fix(None, [blocker()], [{"path": "models/small_cnn.py", "text": ORIGINAL}],
                               self.api, "nrk_abc", policy={"max_params": "500K"})
        req = StubFix.state["requests"][-1]
        self.assertEqual(req["path"], "/api/v1/fix")
        self.assertEqual(req["headers"]["Authorization"], "Bearer nrk_abc")
        self.assertEqual(req["body"]["findings"], [blocker()])
        self.assertEqual(req["body"]["files"][0]["path"], "models/small_cnn.py")
        self.assertEqual(req["body"]["policy"], {"max_params": "500K"})
        self.assertEqual(data["patches"][0]["path"], "models/small_cnn.py")

    def test_a_401_is_a_readable_error(self):
        StubFix.state["mode"] = "401"
        with self.assertRaises(fix.FixError) as ctx:
            fix.request_fix(None, [blocker()], [{"path": "a.py", "text": "x"}], self.api, "nrk_abc")
        self.assertIn("HTTP 401", str(ctx.exception))
        self.assertIn("need an API key", str(ctx.exception))

    def test_a_response_without_patches_is_an_error(self):
        StubFix.state["mode"] = "nopatches"
        with self.assertRaises(fix.FixError):
            fix.request_fix(None, [blocker()], [{"path": "a.py", "text": "x"}], self.api, "nrk_abc")

    def test_unreachable(self):
        with self.assertRaises(fix.FixError) as ctx:
            fix.request_fix(None, [blocker()], [{"path": "a.py", "text": "x"}],
                            "http://127.0.0.1:9", "nrk_abc", timeout=2)
        self.assertIn("could not reach", str(ctx.exception))


# ---------------------------------------------------------------- the existing fix PR

class ExistingFixPrTests(unittest.TestCase):
    def pr(self, number, ref, state="open"):
        return {"number": number, "state": state, "html_url": "https://github.com/o/r/pull/%d" % number,
                "head": {"ref": ref, "repo": {"full_name": "o/r"}}}

    def test_an_open_pr_on_the_branch_is_found_and_updated_not_duplicated(self):
        prs = [self.pr(4, "some-other-branch"), self.pr(7, "neurarch/fix-pr3")]
        found = fix.existing_fix_pr(prs, fix.branch_name(3))
        self.assertIsNotNone(found)
        self.assertEqual(found["number"], 7)

    def test_a_closed_pr_on_the_branch_does_not_count(self):
        prs = [self.pr(7, "neurarch/fix-pr3", state="closed")]
        self.assertIsNone(fix.existing_fix_pr(prs, fix.branch_name(3)))

    def test_a_fix_pr_for_a_different_pull_request_does_not_count(self):
        prs = [self.pr(7, "neurarch/fix-pr9")]
        self.assertIsNone(fix.existing_fix_pr(prs, fix.branch_name(3)))

    def test_no_open_prs_at_all(self):
        self.assertIsNone(fix.existing_fix_pr([], fix.branch_name(3)))
        self.assertIsNone(fix.existing_fix_pr(None, fix.branch_name(3)))

    def test_branch_name_is_derived_from_the_pull_request_number(self):
        self.assertEqual(fix.branch_name(12), "neurarch/fix-pr12")


# ---------------------------------------------------------------- the comment lines

class CommentLineTests(unittest.TestCase):
    def test_success_line_names_the_pr_the_source_and_the_size(self):
        line = fix.success_line(7, "https://github.com/o/r/pull/7",
                                "the exact patch from neurarch-mcp `suggest_fix`", 141354, 141354)
        self.assertIn("[#7](https://github.com/o/r/pull/7)", line)
        self.assertIn("the new plan is legal", line)
        self.assertIn("141.4K -> 141.4K (+0.0%)", line)
        self.assertIn("suggest_fix", line)
        self.assertIn("opened", line)

    def test_success_line_says_updated_when_the_pr_already_existed(self):
        line = fix.success_line(7, "https://x/7", "a proposal from `POST /api/v1/fix`",
                                141354, 150000, updated=True)
        self.assertIn("updated", line)
        self.assertIn("+6.1%", line)

    def test_failure_line_says_what_was_tried_and_that_nothing_was_pushed(self):
        line = fix.failure_line(
            "the exact patch from neurarch-mcp `suggest_fix`: no mechanical fix for these findings")
        self.assertTrue(line.startswith("**No fix was pushed.**"))
        self.assertTrue(line.startswith("**No fix was pushed.** The exact patch"), line)
        self.assertIn("no mechanical fix for these findings.", line)
        self.assertIn("Nothing was pushed", line)
        self.assertIn("10% of this branch's head", line)

    def test_failure_line_can_carry_the_skipped_source(self):
        line = fix.failure_line("nothing mechanical to change",
                                skipped=["The server source was skipped: no api-key is set."])
        self.assertIn("Nothing mechanical to change.", line)
        self.assertIn("no api-key is set", line)

    def test_the_fix_line_goes_above_the_footer_and_replaces_itself(self):
        body = ("<!-- neurarch-bot -->\n## Neurarch plan\n\n### `m.py:M` blocked\n\n"
                "<sub>neurarch-bot v0.3.0.</sub>\n")
        once = fix.comment_with_fix_line(body, "**No fix was pushed.** nothing.")
        self.assertIn(fix.MARKER, once)
        self.assertLess(once.index(fix.MARKER), once.index("<sub>"))
        self.assertTrue(once.startswith("<!-- neurarch-bot -->"))
        twice = fix.comment_with_fix_line(once, "**A fix is open: [#8](u).**")
        self.assertEqual(twice.count(fix.MARKER), 1)
        self.assertNotIn("nothing.", twice)
        self.assertIn("[#8](u)", twice)

    def test_a_comment_with_no_footer_still_gets_the_line(self):
        out = fix.comment_with_fix_line("just a body\n", "**No fix was pushed.** nothing.")
        self.assertIn(fix.MARKER, out)
        self.assertIn("No fix was pushed", out)


# ---------------------------------------------------------------- the PR body

class FixPrBodyTests(unittest.TestCase):
    def body(self):
        return fix.fix_pr_body(
            pr=3, target="models/small_cnn.py:SmallCNN",
            before={"summary": "SmallCNN: will not run (1 blocker)", "params": 141354,
                    "blockers": [blocker("policy: forbidden layer type softmax present")]},
            after={"summary": "SmallCNN: will run", "params": 141354, "blockers": []},
            source="the exact patch from neurarch-mcp `suggest_fix`",
            patches=[{"path": "models/small_cnn.py", "findingTitles": ["Drop the softmax call at the output."]}],
            rounds=1, version="0.3.0",
            notes=["a proposal from `POST /api/v1/fix`: skipped, no key"],
            skipped=["The server source was skipped: no api-key is set."])

    def test_both_plans_are_side_by_side_and_the_source_is_named(self):
        b = self.body()
        self.assertIn("| Before (#3) | SmallCNN: will not run (1 blocker) | 141.4K | 1 blocker |", b)
        self.assertIn("| After (this branch) | SmallCNN: will run | 141.4K | legal |", b)
        self.assertIn("**The exact patch from neurarch-mcp `suggest_fix`**, accepted on round 1 of 3", b)
        self.assertIn("Drop the softmax call at the output.", b)

    def test_it_says_the_fix_pr_gets_no_check_of_its_own(self):
        self.assertIn("does not trigger workflows", self.body())
        self.assertIn("pass a PAT", self.body())

    def test_it_says_what_the_gate_was(self):
        b = self.body()
        self.assertIn("within 10% of #3's own head", b)
        self.assertIn("scratch worktree", b)
        self.assertIn("no api-key is set", b)


class CommitOnlyThePatchedFilesTests(unittest.TestCase):
    """Tracing the model in the scratch worktree writes `__pycache__/*.pyc` next to the source.
    The first fix pull request this bot ever opened carried one, because the commit was
    `git add -A`. The commit is the patch's own paths and nothing else."""

    def test_a_byproduct_of_the_trace_is_not_committed(self):
        import bot  # noqa: PLC0415

        root = tempfile.mkdtemp()
        work, bare = os.path.join(root, "w"), os.path.join(root, "bare.git")
        os.makedirs(os.path.join(work, "models", "__pycache__"))
        subprocess.run(["git", "init", "-q", "--bare", bare], capture_output=True)
        with open(os.path.join(work, "models", "small_cnn.py"), "w") as f:
            f.write(ORIGINAL)
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
            subprocess.run(cmd, cwd=work, capture_output=True)
        with open(os.path.join(work, "models", "small_cnn.py"), "w") as f:
            f.write("line one\nline TWO\nline three\n")
        with open(os.path.join(work, "models", "__pycache__", "small_cnn.cpython-311.pyc"), "wb") as f:
            f.write(b"\x00byte code")

        os.environ["GITHUB_SERVER_URL"] = "file://" + root
        try:
            ok = bot.push_fix_branch(work, work, "bare", "tok", "neurarch/fix-pr4", 4,
                                     "models/small_cnn.py:SmallCNN", ["models/small_cnn.py"])
        finally:
            os.environ.pop("GITHUB_SERVER_URL", None)
        self.assertTrue(ok)
        listed = subprocess.run(["git", "show", "--name-only", "--format=", "neurarch/fix-pr4"],
                                cwd=bare, capture_output=True, text=True).stdout.split()
        self.assertEqual(listed, ["models/small_cnn.py"])


if __name__ == "__main__":
    unittest.main()
