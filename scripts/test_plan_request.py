"""Unit tests for plan_request.py against a stub HTTP server. No torch, no network."""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_request  # noqa: E402

GRAPH = {"id": "g", "name": "TinyCNN", "components": [{"id": "input", "type": "input"}], "connections": []}
BASE = {"id": "b", "name": "TinyCNN", "components": [{"id": "input", "type": "input"}], "connections": []}


def plan_fixture(name, legal=True, summary=None):
    run = {"legal": legal, "blockers": [], "warnings": []}
    if not legal:
        run["blockers"] = [{"title": "Shape problem: multiHeadAttention", "stage": "preflight",
                            "detail": "embed dim (384) must be divisible by numHeads (5)"}]
    summary = summary or ("%s: 9 layers, 60.4K params, %s" % (name, "will run" if legal else "will not run (1 blocker)"))
    plan = {
        "version": 1, "summary": summary,
        "model": {"name": name, "layers": 9, "params": 60426, "inputShape": [3, 32, 32], "outputShape": [10],
                  "fingerprint": "abcd1234"},
        "run": run,
        "cost": {"estCostUsd": 0.04, "estTrainSec": 140.0, "fitsGpu": "A10G (24GB)",
                 "gpuFits": {"T4": True, "A100_40": True, "H100_80": True}, "assumptions": "roofline"},
        "diff": {"base": {"name": name, "layers": 9, "params": 60426}, "added": 0, "removed": 0, "changed": 2,
                 "paramsDelta": 178368, "paramsDeltaPct": 295.2, "memoryDeltaGb": 0.003, "costDeltaUsd": 0.0005,
                 "blastRadius": 4, "lines": ["~ f.4.outChannels 64 -> 256"], "legalityChanged": None},
    }
    markdown = "**%s**\n\n| | |\n|---|---|\n| Layers / params | 9 / 60.4K (+178.4K, +295%%) |\n| Est. training | ~$0.04 / 3m |\n" % summary
    text = "Plan: %s   9 layers\n" % name
    return {"plan": plan, "markdown": markdown, "text": text}


class StubHandler(BaseHTTPRequestHandler):
    state = {"requests": [], "mode": "ok"}

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        self.state["requests"].append({
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": json.loads(raw.decode("utf-8")),
        })
        mode = self.state["mode"]
        if mode == "429":
            payload = json.dumps({"error": "rate limit: 60 requests per hour per IP"}).encode("utf-8")
            self.send_response(429)
        elif mode == "500":
            payload = b"boom"
            self.send_response(500)
        elif mode == "notjson":
            payload = b"<html>"
            self.send_response(200)
        else:
            payload = json.dumps(plan_fixture("WideCNN")).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence
        pass


class PlanRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), StubHandler)
        cls.api = "http://127.0.0.1:%d" % cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        StubHandler.state["requests"] = []
        StubHandler.state["mode"] = "ok"

    def last_request(self):
        return StubHandler.state["requests"][-1]

    def test_request_body_fields(self):
        source = plan_request.build_source("models/cnn.py:SmallCNN", "1,3,32,32", pr=7)
        resp = plan_request.request_plan(GRAPH, BASE, source, api=self.api)
        req = self.last_request()
        self.assertEqual(req["path"], "/api/v1/plan")
        body = req["body"]
        self.assertEqual(body["model"], GRAPH)
        self.assertEqual(body["base"], BASE)
        self.assertIs(body["share"], False)
        self.assertEqual(body["source"], {
            "kind": "bot", "target": "models/cnn.py:SmallCNN", "input": "1,3,32,32",
            "tool": "neurarch-bot/%s" % plan_request.__version__, "pr": 7,
        })
        self.assertEqual(req["headers"]["User-Agent"], "neurarch-bot/%s" % plan_request.__version__)
        self.assertEqual(resp["plan"]["model"]["name"], "WideCNN")

    def test_no_base_means_no_base_field(self):
        source = plan_request.build_source("m.py:M", "1,8:long", pr=None)
        plan_request.request_plan(GRAPH, None, source, api=self.api)
        body = self.last_request()["body"]
        self.assertNotIn("base", body)
        self.assertNotIn("pr", body["source"])

    def test_key_header_present_and_absent(self):
        source = plan_request.build_source("m.py:M", "1,3,32,32", pr=1)
        plan_request.request_plan(GRAPH, None, source, api=self.api, api_key="nrk_test123")
        self.assertEqual(self.last_request()["headers"]["Authorization"], "Bearer nrk_test123")
        plan_request.request_plan(GRAPH, None, source, api=self.api, api_key=None)
        self.assertNotIn("Authorization", self.last_request()["headers"])
        plan_request.request_plan(GRAPH, None, source, api=self.api, api_key="")
        self.assertNotIn("Authorization", self.last_request()["headers"])

    def test_429_names_the_api_key_input(self):
        StubHandler.state["mode"] = "429"
        source = plan_request.build_source("m.py:M", "1,3,32,32", pr=1)
        with self.assertRaises(plan_request.PlanRequestError) as ctx:
            plan_request.request_plan(GRAPH, None, source, api=self.api)
        msg = str(ctx.exception)
        self.assertIn("rate-limited", msg)
        self.assertIn("60 requests per hour per IP", msg)
        self.assertIn("api-key", msg)

    def test_500_and_non_json(self):
        source = plan_request.build_source("m.py:M", "1,3,32,32", pr=1)
        StubHandler.state["mode"] = "500"
        with self.assertRaises(plan_request.PlanRequestError) as ctx:
            plan_request.request_plan(GRAPH, None, source, api=self.api)
        self.assertIn("HTTP 500", str(ctx.exception))
        StubHandler.state["mode"] = "notjson"
        with self.assertRaises(plan_request.PlanRequestError) as ctx:
            plan_request.request_plan(GRAPH, None, source, api=self.api)
        self.assertIn("not JSON", str(ctx.exception))

    def test_unreachable(self):
        source = plan_request.build_source("m.py:M", "1,3,32,32", pr=1)
        with self.assertRaises(plan_request.PlanRequestError) as ctx:
            plan_request.request_plan(GRAPH, None, source, api="http://127.0.0.1:9", timeout=2)
        self.assertIn("could not reach", str(ctx.exception))

    def test_write_plan(self):
        resp = plan_fixture("WideCNN")
        with tempfile.TemporaryDirectory() as d:
            paths = plan_request.write_plan(resp, os.path.join(d, "out"))
            with open(paths["json"]) as f:
                self.assertEqual(json.load(f)["summary"], resp["plan"]["summary"])
            with open(paths["md"]) as f:
                self.assertEqual(f.read(), resp["markdown"])
            with open(paths["txt"]) as f:
                self.assertEqual(f.read(), resp["text"])


class CommentAssemblyTests(unittest.TestCase):
    def entries(self):
        ok = plan_fixture("WideCNN")
        bad = plan_fixture("TinyGPT", legal=False)
        return [
            {"target": "models/small_cnn.py:SmallCNN", "input": "1,3,32,32", "status": "planned",
             "plan": ok["plan"], "markdown": ok["markdown"], "legal": True, "blockers": 0, "warnings": 0},
            {"target": "models/tiny_gpt.py:TinyGPT", "input": "1,32:long", "status": "planned",
             "plan": bad["plan"], "markdown": bad["markdown"], "legal": False, "blockers": 1, "warnings": 0},
        ]

    def test_marker_first_line_and_sections_in_order(self):
        body = plan_request.assemble_comment(self.entries(), version="0.1.0", run_url="https://x/run/1")
        lines = body.split("\n")
        self.assertEqual(lines[0], plan_request.MARKER)
        i_cnn = body.index("### `models/small_cnn.py:SmallCNN`")
        i_gpt = body.index("### `models/tiny_gpt.py:TinyGPT`")
        self.assertLess(i_cnn, i_gpt)
        self.assertIn(":white_check_mark: will run", body[i_cnn:i_gpt])
        self.assertIn(":no_entry: blocked", body[i_gpt:])
        self.assertIn("https://x/run/1", body)
        self.assertIn("never weights or data", body)
        self.assertIn("neurarch-bot v0.1.0", body)

    def test_summary_bold_once_then_server_markdown(self):
        entries = self.entries()
        body = plan_request.assemble_comment(entries)
        for e in entries:
            summary = "**%s**" % e["plan"]["summary"]
            self.assertEqual(body.count(summary), 1, summary)
            section = body[body.index("### `%s`" % e["target"]):]
            self.assertLess(section.index(summary), section.index("| Layers / params |"))
        self.assertIn("(+178.4K, +295%)", body)

    def test_trace_failure_section(self):
        entry = {"target": "models/tiny_gpt.py:TinyGPT", "input": "1,32:long", "status": "trace_failed",
                 "error": "Traceback (most recent call last):\n  ...\nAssertionError: embed_dim must be divisible by num_heads",
                 "lint_markdown": "### neurarch-lint\n\n:no_entry: **head-dim-divisibility** (block)",
                 "lint_findings": [{"severity": "block"}], "legal": False, "blockers": 1, "warnings": 0}
        body = plan_request.assemble_comment([entry])
        self.assertIn("could not trace", body)
        self.assertIn("needs fixing", body)
        self.assertIn("AssertionError: embed_dim must be divisible by num_heads", body)
        self.assertIn("head-dim-divisibility", body)

    def test_has_blocker(self):
        self.assertFalse(plan_request.has_blocker(plan_fixture("A")["plan"]))
        self.assertTrue(plan_request.has_blocker(plan_fixture("A", legal=False)["plan"]))
        self.assertTrue(plan_request.has_blocker({"run": {"legal": True, "blockers": [{"title": "x"}]}}))
        self.assertFalse(plan_request.has_blocker({}))


if __name__ == "__main__":
    unittest.main()


class AffectedTests(unittest.TestCase):
    def setUp(self):
        import bot
        self.bot = bot
        self.gpt = {"target": "models/tiny_gpt.py:TinyGPT", "file": "models/tiny_gpt.py"}
        self.cnn = {"target": "models/small_cnn.py:SmallCNN", "file": "models/small_cnn.py"}
        self.listed = [self.gpt["file"], self.cnn["file"]]

    def test_own_file(self):
        self.assertTrue(self.bot.is_affected(self.cnn, ["models/small_cnn.py"], self.listed))
        self.assertFalse(self.bot.is_affected(self.gpt, ["models/small_cnn.py"], self.listed))

    def test_helper_under_directory(self):
        self.assertTrue(self.bot.is_affected(self.gpt, ["models/layers/attention.py"], self.listed))
        self.assertTrue(self.bot.is_affected(self.cnn, ["models/__init__.py"], self.listed))

    def test_unrelated(self):
        self.assertFalse(self.bot.is_affected(self.gpt, ["README.md", ".neurarch.yml", "train/run.py"], self.listed))

    def test_config_parse(self):
        cfg = {"models": [{"path": "models/tiny_gpt.py:TinyGPT", "input": "1,32:long", "depth": 2,
                           "extra_inputs": ["1,32:long"]}]}
        m = self.bot.parse_models(cfg)[0]
        self.assertEqual(m["file"], "models/tiny_gpt.py")
        self.assertEqual(m["input"], "1,32:long")
        self.assertEqual(m["depth"], 2)
        self.assertEqual(m["extra_inputs"], ["1,32:long"])
        with self.assertRaises(SystemExit):
            self.bot.parse_models({"models": [{"path": "models/x.py"}]})
        with self.assertRaises(SystemExit):
            self.bot.parse_models({"models": [{"path": "models/x.py:X"}]})
