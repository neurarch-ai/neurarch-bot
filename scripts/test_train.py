"""Unit tests for train.py against a stub HTTP server. No torch, no network, no GPU.

The stub is the shape of the real endpoint, and the fixtures are copied from a real run and a
real refusal: a completed 3-epoch MNIST job (34s, $0.0075, ledger written) and a job that
crashed in the worker with `embed_dim must be divisible by num_heads`.
"""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402

GRAPH = {"id": "g", "name": "MnistNet", "components": [{"id": "input", "type": "input"}],
         "connections": []}

BLOCKED_PLAN = {
    "version": 1,
    "summary": "SmallCNN: 21 layers, 1.1M params, will not run (1 blocker)",
    "model": {"name": "SmallCNN", "layers": 21, "params": 1107146},
    "run": {"legal": False, "warnings": [], "blockers": [
        {"title": "policy: max_params 500K exceeded (1.1M)", "stage": "policy",
         "detail": "max_params: 500K",
         "fix": "Reduce the model by 620.2K parameters or raise max_params."}]},
}

EPOCHS = [
    {"epoch": 1, "valAcc": 0.9712, "valLoss": 0.0921, "trainAcc": 0.9410, "trainLoss": 0.1904},
    {"epoch": 2, "valAcc": 0.9820, "valLoss": 0.0603, "trainAcc": 0.9799, "trainLoss": 0.0651},
    {"epoch": 3, "valAcc": 0.9865, "valLoss": 0.0477, "trainAcc": 0.9858, "trainLoss": 0.0456},
]


def job_fixture(status, epochs=0, **extra):
    job = {"job_id": "job_1788488972586_eoblzo", "status": status, "mode": "gpu",
           "dataset": "mnist", "epochs": EPOCHS[:epochs], "metrics": [],
           "wall_sec": None, "est_cost_usd": None, "created_at": "2026-09-04T02:29:32.586+00:00",
           "completed_at": None, "fingerprint": "da073440", "config": {"epochs": 3}}
    if status == "completed":
        job.update({"wall_sec": 34, "est_cost_usd": 0.0075, "ledger": "written",
                    "completed_at": "2026-09-04T02:30:06.000+00:00",
                    "metrics": [{"name": "valAcc", "value": 0.9865},
                                {"name": "valLoss", "value": 0.0477}]})
    job.update(extra)
    return job


class StubHandler(BaseHTTPRequestHandler):
    state = {"requests": [], "post": "ok", "polls": []}

    def _send(self, code, payload):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.state["requests"].append({
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": json.loads(self.rfile.read(length).decode("utf-8")),
        })
        mode = self.state["post"]
        if mode == "blocked":
            return self._send(422, {"error": "blocked", "plan": BLOCKED_PLAN,
                                    "text": "Plan: SmallCNN   will not run"})
        if mode == "nogpu":
            return self._send(503, {"error": "gpu_not_configured",
                                    "message": "No managed GPU worker is configured."})
        if mode == "413":
            return self._send(413, {"error": "Model too large (900 components; max 800)."})
        if mode == "html":
            self.send_response(500)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", "6")
            self.end_headers()
            return self.wfile.write(b"<html>")
        return self._send(200, {"job_id": "job_1788488972586_eoblzo", "mode": "gpu",
                                "backend": "modal",
                                "config": {"epochs": 20, "batchSize": 64, "lr": 0.001,
                                           "optimizer": "adamw"},
                                "plan": {"cost": {"estCostUsd": 0.0075}},
                                "text": "Plan: MnistNet"})

    def do_GET(self):
        self.state["requests"].append({"path": self.path,
                                       "headers": {k: v for k, v in self.headers.items()}})
        queue = self.state["polls"]
        if not queue:
            return self._send(404, {"error": "No such job under this key."})
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return self._send(200, item)

    def log_message(self, *args):  # silence
        pass


class ServerCase(unittest.TestCase):
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
        StubHandler.state["post"] = "ok"
        StubHandler.state["polls"] = []

    def last(self):
        return StubHandler.state["requests"][-1]


# ---------------------------------------------------------------- the command

class CommandTests(unittest.TestCase):
    def test_recognised_only_on_the_first_line_and_anchored(self):
        self.assertTrue(train.is_command("/neurarch train"))
        self.assertTrue(train.is_command("  /neurarch train epochs=3\n\nplease"))
        self.assertTrue(train.is_command("/neurarch train\r\nsecond line"))
        # Quoting the command, or writing about it, must never book a GPU.
        self.assertFalse(train.is_command("> /neurarch train"))
        self.assertFalse(train.is_command("try `/neurarch train` on this"))
        self.assertFalse(train.is_command("/neurarch trainer"))
        self.assertFalse(train.is_command("/neurarch plan"))
        self.assertFalse(train.is_command(""))
        self.assertFalse(train.is_command(None))

    def test_no_arguments_is_valid(self):
        self.assertEqual(train.parse_command("/neurarch train"), ({}, None))

    def test_every_known_argument(self):
        args, err = train.parse_command(
            "/neurarch train epochs=5 dataset=CIFAR10 gpu=A10G model=models/cnn.py:SmallCNN "
            "batch=128 lr=3e-4 optimizer=SGD")
        self.assertIsNone(err)
        self.assertEqual(args["epochs"], "5")
        self.assertEqual(args["dataset"], "CIFAR10")
        self.assertEqual(args["gpu"], "A10G")
        self.assertEqual(args["model"], "models/cnn.py:SmallCNN")
        self.assertEqual(train.config_from_args(args),
                         {"epochs": 5, "batchSize": 128, "lr": 0.0003, "optimizer": "sgd"})

    def test_junk_is_refused_rather_than_ignored(self):
        # The whole point: `epoch=3` silently training the default is the failure this prevents.
        for body, needle in [
            ("/neurarch train epoch=3", "`epoch` is not an argument"),
            ("/neurarch train 3", "is not `key=value`"),
            ("/neurarch train epochs=", "has no value"),
            ("/neurarch train epochs=three", "not a whole number of epochs"),
            ("/neurarch train epochs=0", "not a whole number of epochs"),
            ("/neurarch train epochs=-2", "not a whole number of epochs"),
            ("/neurarch train batch=big", "not a whole number"),
            ("/neurarch train lr=fast", "not a positive number"),
            ("/neurarch train lr=0", "not a positive number"),
            ("/neurarch train dataset=imagenet", "is not one of"),
            ("/neurarch train optimizer=rmsprop", "is not one of"),
            ("/neurarch train model=SmallCNN", "file.py:ClassOrFactory"),
            ("/neurarch train epochs=3 epochs=4", "given twice"),
        ]:
            args, err = train.parse_command(body)
            self.assertEqual(args, {}, body)
            self.assertIsNotNone(err, body)
            self.assertIn(needle, err, body)

    def test_arguments_only_come_from_the_command_line(self):
        args, err = train.parse_command("/neurarch train epochs=2\nepochs=99 dataset=cifar10")
        self.assertIsNone(err)
        self.assertEqual(args, {"epochs": "2"})

    def test_config_carries_only_what_was_asked_for(self):
        self.assertEqual(train.config_from_args({}), {})
        self.assertEqual(train.config_from_args({"epochs": "3"}), {"epochs": 3})


# ---------------------------------------------------------------- the gates

class AssociationTests(unittest.TestCase):
    def test_write_access_may_spend(self):
        for assoc in ("OWNER", "MEMBER", "COLLABORATOR", "owner", " Collaborator "):
            self.assertTrue(train.may_spend(assoc), assoc)

    def test_everyone_else_may_not(self):
        for assoc in ("CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "MANNEQUIN",
                      "NONE", "", None, "OWNERS", "COLLABORATOR_"):
            self.assertFalse(train.may_spend(assoc), assoc)

    def test_the_refusal_names_who_and_says_nothing_was_submitted(self):
        body = train.access_refusal("someone", "CONTRIBUTOR")
        self.assertIn("@someone", body)
        self.assertIn("`CONTRIBUTOR`", body)
        self.assertIn("Nothing was submitted.", body)
        self.assertIn("write access", body)


class ChooseModelTests(unittest.TestCase):
    MODELS = [{"target": "models/small_cnn.py:SmallCNN"}, {"target": "models/tiny_gpt.py:TinyGPT"}]

    def test_one_listed_model_needs_no_argument(self):
        model, err = train.choose_model(self.MODELS[:1], None)
        self.assertIsNone(err)
        self.assertEqual(model["target"], "models/small_cnn.py:SmallCNN")

    def test_several_and_none_named_is_a_question_not_a_guess(self):
        model, err = train.choose_model(self.MODELS, None)
        self.assertIsNone(model)
        self.assertIn("say which one", err)
        self.assertIn("models/tiny_gpt.py:TinyGPT", err)

    def test_named_by_full_target_or_by_class(self):
        self.assertEqual(train.choose_model(self.MODELS, "models/tiny_gpt.py:TinyGPT")[0]["target"],
                         "models/tiny_gpt.py:TinyGPT")
        self.assertEqual(train.choose_model(self.MODELS, "TinyGPT")[0]["target"],
                         "models/tiny_gpt.py:TinyGPT")

    def test_unknown_model_lists_what_there_is(self):
        model, err = train.choose_model(self.MODELS, "models/other.py:Other")
        self.assertIsNone(model)
        self.assertIn("does not list", err)
        self.assertIn("models/small_cnn.py:SmallCNN", err)

    def test_no_models_at_all(self):
        model, err = train.choose_model([], None)
        self.assertIsNone(model)
        self.assertIn("nothing to train", err)


# ---------------------------------------------------------------- the wire

class SubmitTests(ServerCase):
    def test_body_key_and_source(self):
        status, data, _ = train.submit_train(
            GRAPH, "mnist", self.api, "nrk_test", config={"epochs": 3}, gpu="T4",
            source=train.build_source("models/cnn.py:SmallCNN", "o/r", 2, "0.4.0"))
        self.assertEqual(status, 200)
        self.assertEqual(data["job_id"], "job_1788488972586_eoblzo")
        req = self.last()
        self.assertEqual(req["path"], "/api/v1/train")
        self.assertEqual(req["headers"]["Authorization"], "Bearer nrk_test")
        self.assertEqual(req["body"]["model"], GRAPH)
        self.assertEqual(req["body"]["dataset"], "mnist")
        self.assertEqual(req["body"]["config"], {"epochs": 3})
        self.assertEqual(req["body"]["gpu"], "T4")
        self.assertEqual(req["body"]["source"], {
            "kind": "bot", "via": "neurarch-bot/0.4.0", "target": "models/cnn.py:SmallCNN",
            "repo": "o/r", "pr": 2})

    def test_absent_optionals_are_absent_on_the_wire(self):
        train.submit_train(GRAPH, "mnist", self.api, "nrk_test")
        body = self.last()["body"]
        for key in ("config", "gpu", "source"):
            self.assertNotIn(key, body)

    def test_blocked_comes_back_as_data_not_an_exception(self):
        StubHandler.state["post"] = "blocked"
        status, data, _ = train.submit_train(GRAPH, "mnist", self.api, "nrk_test")
        self.assertEqual(status, 422)
        self.assertEqual(data["error"], "blocked")
        self.assertFalse(data["plan"]["run"]["legal"])

    def test_unreachable_is_the_one_exception(self):
        with self.assertRaises(train.TrainRequestError):
            train.submit_train(GRAPH, "mnist", "http://127.0.0.1:1", "nrk_test", timeout=2)

    def test_a_non_json_error_body_still_yields_a_message(self):
        StubHandler.state["post"] = "html"
        status, data, message = train.submit_train(GRAPH, "mnist", self.api, "nrk_test")
        self.assertEqual(status, 500)
        self.assertEqual(data, {})
        self.assertIn("500", message)


class PollTests(ServerCase):
    def test_poll_reads_the_job(self):
        StubHandler.state["polls"] = [job_fixture("running", epochs=1)]
        job = train.poll_job("job_1788488972586_eoblzo", self.api, "nrk_test")
        self.assertEqual(job["status"], "running")
        self.assertIn("job=job_1788488972586_eoblzo", self.last()["path"])
        self.assertEqual(self.last()["headers"]["Authorization"], "Bearer nrk_test")

    def test_someone_elses_job_is_an_error_not_a_result(self):
        StubHandler.state["polls"] = []
        with self.assertRaises(train.TrainRequestError):
            train.poll_job("job_1788488972586_eoblzo", self.api, "nrk_test")

    def test_follow_walks_queued_running_completed_and_updates_only_on_change(self):
        StubHandler.state["polls"] = [
            job_fixture("queued"), job_fixture("running", epochs=1),
            job_fixture("running", epochs=1),  # nothing new: no comment edit
            job_fixture("running", epochs=2), job_fixture("completed", epochs=3),
        ]
        seen = []
        slept = []
        job, timed_out = train.follow_job(
            "job_1788488972586_eoblzo", self.api, "nrk_test",
            on_change=lambda j: seen.append((j["status"], len(j["epochs"]))),
            timeout_min=10, poll_seconds=20, sleep=slept.append, clock=lambda: len(slept) * 20.0)
        self.assertFalse(timed_out)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(seen, [("queued", 0), ("running", 1), ("running", 2), ("completed", 3)])
        self.assertEqual(slept, [20, 20, 20, 20])

    def test_a_failed_poll_does_not_end_the_run(self):
        calls = {"n": 0}

        def flaky(job_id, api, key):
            calls["n"] += 1
            if calls["n"] == 1:
                raise train.TrainRequestError("connection reset")
            return job_fixture("completed", epochs=3)

        job, timed_out = train.follow_job(
            "job_x", self.api, "k", on_change=lambda j: None, timeout_min=10, poll_seconds=1,
            sleep=lambda s: None, clock=lambda: 0.0, poll=flaky)
        self.assertFalse(timed_out)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(calls["n"], 2)

    def test_timeout_returns_the_last_state_and_says_so(self):
        ticks = {"t": 0.0}

        def clock():
            ticks["t"] += 30.0
            return ticks["t"]

        job, timed_out = train.follow_job(
            "job_x", self.api, "k", on_change=lambda j: None, timeout_min=1, poll_seconds=20,
            sleep=lambda s: None, clock=clock,
            poll=lambda *a: job_fixture("running", epochs=1))
        self.assertTrue(timed_out)
        self.assertEqual(job["status"], "running")
        body = train.run_comment("job_x", "m.py:M", "mnist", {"epochs": 3}, None, "0.4.0",
                                 "www.neurarch.com", job=job, timed_out=True)
        self.assertIn("timeout-min", body)
        self.assertIn("is not cancelled", body)
        self.assertIn("GET /api/v1/train?job=job_x", body)


# ---------------------------------------------------------------- the comments

class RefusalBodyTests(unittest.TestCase):
    def test_blocked_by_the_server_names_the_blocker_and_the_spend(self):
        body = train.blocked_comment("models/small_cnn.py:SmallCNN", BLOCKED_PLAN,
                                     "Plan: SmallCNN   will not run", "0.4.0", "www.neurarch.com")
        self.assertTrue(body.startswith(train.MARKER))
        self.assertIn("nothing was submitted and nothing was spent", body)
        self.assertIn("policy: max_params 500K exceeded (1.1M)", body)
        self.assertIn("Fix: Reduce the model by 620.2K parameters", body)
        self.assertIn("Plan: SmallCNN   will not run", body)
        # A refusal is the endpoint working, and reads that way rather than as a failure.
        self.assertIn("That ordering is the point", body)
        self.assertNotIn("<!-- neurarch-bot -->", body)

    def test_blocked_by_the_house_rules_says_the_server_never_saw_them(self):
        body = train.blocked_comment("m.py:M", BLOCKED_PLAN, "", "0.4.0", "h", source="policy")
        self.assertIn("`.neurarch.yml`", body)
        self.assertIn("never sees it", body)
        self.assertIn("nothing was submitted and nothing was spent", body)

    def test_no_gpu_refuses_to_offer_a_simulation(self):
        body = train.gpu_unconfigured_comment("No managed GPU worker is configured.",
                                              "0.4.0", "www.neurarch.com")
        self.assertIn("Managed GPU is not configured", body)
        self.assertIn("simulated curve is not a training run", body)
        self.assertIn("Nothing was submitted", body)
        self.assertIn("No managed GPU worker is configured.", body)

    def test_other_errors_quote_the_message(self):
        body = train.error_comment(413, "Model too large (900 components; max 800).",
                                   "0.4.0", "www.neurarch.com")
        self.assertIn("HTTP 413", body)
        self.assertIn("Model too large (900 components; max 800).", body)
        self.assertIn("Nothing was submitted", body)

    def test_no_key(self):
        body = train.no_key_refusal()
        self.assertIn("needs an API key", body)
        self.assertIn("NEURARCH_API_KEY", body)
        self.assertIn("Nothing was submitted", body)


class RunCommentTests(unittest.TestCase):
    def body(self, job=None, **kw):
        return train.run_comment("job_1788488972586_eoblzo", "models/mnist.py:MnistNet", "mnist",
                                 {"epochs": 3, "batchSize": 64, "lr": 0.001, "optimizer": "adamw"},
                                 {"cost": {"estCostUsd": 0.0075}}, "0.4.0", "www.neurarch.com",
                                 job=job, **kw)

    def test_the_first_comment_carries_the_job_marker_config_and_estimate(self):
        body = self.body()
        self.assertTrue(body.startswith("<!-- neurarch-train:job_1788488972586_eoblzo -->"))
        self.assertIn("`job_1788488972586_eoblzo`", body)
        self.assertIn("3 epochs, batch 64, lr 0.001, adamw", body)
        self.assertIn("$0.0075 (roofline, before the run)", body)
        self.assertIn("This comment is updated as epochs arrive.", body)

    def test_a_clamped_config_is_said_out_loud(self):
        note = train.clamp_note({"epochs": 200}, {"epochs": 20, "batchSize": 64, "lr": 0.001,
                                                  "optimizer": "adamw"})
        self.assertIn("epochs 200 → 20", note)
        self.assertIn(note, self.body(clamped=note))
        # Nothing changed, nothing said.
        self.assertEqual(train.clamp_note({"epochs": 3}, {"epochs": 3, "batchSize": 64}), "")

    def test_running_shows_the_epochs_so_far(self):
        body = self.body(job=job_fixture("running", epochs=2))
        self.assertIn("| Progress | epoch 2 of 3 |", body)
        self.assertIn("| 1 | 0.1904 | 94.10% | 0.0921 | 97.12% |", body)
        self.assertNotIn("| 3 |", body)

    def test_completed_carries_the_table_the_final_metric_cost_and_the_ledger(self):
        body = self.body(job=job_fixture("completed", epochs=3))
        self.assertIn("finished", body)
        # A finished run is not "epoch 3 of 20": it ran three and stopped.
        self.assertIn("| Epochs | 3 |", body)
        self.assertNotIn("Progress", body)
        self.assertIn("| 3 | 0.0456 | 98.58% | 0.0477 | 98.65% |", body)
        self.assertIn("**val acc 98.65%**", body)
        self.assertIn("| Wall clock | 34s |", body)
        self.assertIn("$0.0075 (measured from wall clock", body)
        self.assertIn("next `/api/v1/plan` for this structure will quote this run", body)
        # The curve is drawn, and it is never an image: GitHub strips a data: URI src.
        self.assertIn("`val acc  ", body)
        self.assertNotIn("data:image/svg+xml", body)
        self.assertNotIn("<svg", body)

    def test_a_failed_run_reports_the_trainer_and_not_the_check(self):
        job = job_fixture("error", epochs=0,
                          error_detail="Traceback (most recent call last):\n"
                                       "  File \"/tmp/x.py\", line 110, in __init__\n"
                                       "AssertionError: embed_dim must be divisible by num_heads")
        body = self.body(job=job)
        self.assertIn("the run failed", body)
        self.assertIn("the check is not red because of it", body)
        self.assertIn("AssertionError: embed_dim must be divisible by num_heads", body)
        self.assertNotIn("val acc", body)

    def test_a_simulated_run_is_never_reported_as_a_measurement(self):
        body = self.body(job=job_fixture("simulation", epochs=3))
        self.assertIn("not a training run", body)
        self.assertIn("nothing was written to the ledger", body)

    def test_a_skipped_ledger_says_the_next_plan_will_not_quote_it(self):
        job = job_fixture("completed", epochs=3)
        job["ledger"] = "skipped:no_metrics"
        line = train.ledger_line(job)
        self.assertIn("skipped:no_metrics", line)
        self.assertIn("will not quote this run", line)


class CurveTests(unittest.TestCase):
    def test_sparkline_spans_min_to_max(self):
        self.assertEqual(train.sparkline([0.0, 1.0]), "▁█")
        self.assertEqual(train.sparkline([1.0, 1.0, 1.0]), "▁▁▁")
        self.assertEqual(train.sparkline([]), "")
        self.assertEqual(len(train.sparkline([0.97, 0.982, 0.9865])), 3)

    def test_utilisation_and_eval_rows_are_not_epochs(self):
        job = {"epochs": EPOCHS + [{"util": True, "gpu": 0.8}, {"eval": True, "valAcc": 0.99}]}
        self.assertEqual(len(train.epoch_rows(job)), 3)

    def test_no_epochs_draws_nothing(self):
        self.assertEqual(train.render_curve({"epochs": []}), "")

    def test_error_is_trimmed_from_the_end(self):
        text = "\n".join("line %d" % i for i in range(40))
        trimmed = train.trim_error(text, lines=3)
        self.assertEqual(trimmed, "line 37\nline 38\nline 39")
        self.assertEqual(train.trim_error(None), "no output")


class UpsertTests(unittest.TestCase):
    def test_the_job_marker_never_matches_the_plan_comment(self):
        comments = [
            {"id": 1, "body": "<!-- neurarch-bot -->\n## Neurarch plan"},
            {"id": 2, "body": "unrelated"},
            {"id": 3, "body": "<!-- neurarch-train -->\n## Neurarch training"},
            {"id": 4, "body": "<!-- neurarch-train:job_a -->\n## Neurarch training run"},
            {"id": 5, "body": "<!-- neurarch-train:job_b -->\n## Neurarch training run"},
        ]
        self.assertEqual(train.find_comment(comments, train.JOB_MARKER % "job_a")["id"], 4)
        self.assertEqual(train.find_comment(comments, train.JOB_MARKER % "job_b")["id"], 5)
        self.assertIsNone(train.find_comment(comments, train.JOB_MARKER % "job_c"))
        # A run's marker is a prefix of nothing else, so the plan comment is never edited.
        self.assertIsNone(train.find_comment(
            [{"id": 1, "body": "<!-- neurarch-bot -->"}], train.JOB_MARKER % "job_a"))


class FormatTests(unittest.TestCase):
    def test_money_keeps_the_cents_a_short_run_actually_costs(self):
        self.assertEqual(train.fmt_money(0.0075), "$0.0075")
        self.assertEqual(train.fmt_money(0.42), "$0.42")
        self.assertEqual(train.fmt_money(None), "?")

    def test_duration(self):
        self.assertEqual(train.fmt_duration(34), "34s")
        self.assertEqual(train.fmt_duration(125), "2m 05s")
        self.assertEqual(train.fmt_duration(7500), "2h 05m")


if __name__ == "__main__":
    unittest.main()
