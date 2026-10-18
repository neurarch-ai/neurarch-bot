"""Tests for the job half of the comment: memory first, against the card the job reserves.

The rendering and the config reading are pure and run anywhere. The sizing runs the vendored
plan bundle under node and is skipped when node is absent (it never is on a GitHub runner).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bot  # noqa: E402
import job  # noqa: E402
import plan_request  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "tiny_gpt.neurarch.json")
HAS_NODE = shutil.which("node") is not None


def sizing(head_fits=False, base_fits=True, card=True, max_batch=16, smallest="L40S (48GB)"):
    """A size_job.mjs answer with the shape of the demo: a job that doubled its context."""
    head = {"params": 135_800_000, "legal": True, "batch": 32, "seqLen": 2048, "attention": "sdpa",
            "peakGb": 38.4 if not head_fits else 20.6, "footprintGb": 2.7,
            "activationGb": 35.7 if not head_fits else 17.9, "needGb": 44.2 if not head_fits else 23.7,
            "fits": head_fits if card else None, "maxBatch": max_batch if card else None,
            "smallestFitGpu": smallest, "smallestFitGb": 48 if smallest == "L40S (48GB)" else 24,
            "twoCardSplit": None, "costUsd": 16.34, "trainSec": 30240, "costCard": smallest if not head_fits else "A10G (24GB)",
            "epochs": 3, "samples": 400000, "samplesAssumed": False}
    base = {"params": 135_000_000, "legal": True, "batch": 32, "seqLen": 1024, "peakGb": 20.6,
            "footprintGb": 2.7, "activationGb": 17.9, "needGb": 23.7, "fits": base_fits if card else None,
            "maxBatch": 33}
    return {"ok": True, "headroom": 1.15, "card": {"name": "A10G (24GB)", "gb": 24} if card else None,
            "cardError": None, "head": head, "base": base, "baseJob": {"batch": 32, "seq_len": 1024},
            "historyLine": None}


def planned_entry(sizing_answer, job_spec=None, legal=True):
    plan = {
        "summary": "DocLM: 113 layers, 135.8M params, will run, fits a T4 at batch 32",
        "model": {"name": "DocLM", "layers": 113, "params": 135_800_000},
        "run": {"legal": legal, "warnings": [],
                "blockers": [] if legal else [{"title": "Shape problem: linear", "detail": "in 768 != 512"}]},
        "cost": {}, "policy": None,
        "diff": {"base": {"name": "DocLM"}, "added": 0, "removed": 0, "changed": 1, "paramsDelta": 786_432,
                 "blastRadius": 112, "lines": ["~ pos_emb.vocabSize 1024 -> 2048"], "legalityChanged": None},
    }
    return {"target": "models/doc_lm.py:DocLM", "input": "1,32:long", "status": "planned",
            "legal": legal, "blockers": 0 if legal else 1, "warnings": 0, "plan": plan,
            "markdown": "**%s**\n\n| GPU fit | peak ~11.2 GB at batch 32 |" % plan["summary"],
            "sizing": sizing_answer, "job": job_spec or {"gpu": "A10G", "batch": 32, "seq_len": 2048},
            "job_problems": []}


class Render(unittest.TestCase):
    def test_the_section_leads_with_memory_then_structure(self):
        body = plan_request.render_entry(planned_entry(sizing()))
        lines = [l for l in body.splitlines() if l.strip()]
        self.assertIn(":no_entry: no longer fits A10G 24 GB", lines[0])
        self.assertTrue(lines[1].startswith("**Peak training memory: 38.4 GB** at batch 32 x 2,048 tokens"))
        self.assertIn("**Reserved card: A10G 24 GB. Does not fit**", lines[2])
        self.assertIn("**vs base:** +17.8 GB", lines[3])
        self.assertIn("**no longer fits A10G 24 GB**", lines[3])
        self.assertIn("**Largest batch that fits A10G 24 GB:** 16 at 2,048 tokens", lines[4])
        self.assertIn("Batch 16 with 2 gradient-accumulation steps keeps the effective batch at 32", lines[4])
        self.assertIn("**Smallest card that holds it:** L40S 48 GB", lines[5])
        self.assertIn("**Cost:** about $16.34", lines[6])
        self.assertIn("priced on the card that holds it", lines[6])
        structural = body.index("**Structural checks.**")
        self.assertGreater(structural, body.index("**Cost:**"))
        self.assertIn("~ pos_emb.vocabSize 1024 -> 2048", body[structural:])

    def test_the_server_memory_table_does_not_contradict_the_job(self):
        # The hosted plan sizes at its own default batch; with the job sized, that table is gone.
        body = plan_request.render_entry(planned_entry(sizing()))
        self.assertNotIn("peak ~11.2 GB at batch 32", body)
        self.assertNotIn("fits a T4 at batch 32", body)

    def test_a_job_that_fits_says_so_and_names_a_smaller_card(self):
        s = sizing(head_fits=True, smallest="T4 (16GB)")
        s["head"]["smallestFitGb"] = 16
        body = plan_request.render_entry(planned_entry(s))
        self.assertIn(":white_check_mark: fits A10G 24 GB", body.splitlines()[0])
        self.assertIn("**Reserved card: A10G 24 GB. Fits**", body)
        self.assertIn("T4 16 GB, smaller than the 24 GB this job reserves", body)
        self.assertNotIn("gradient-accumulation", body)

    def test_same_peak_reads_as_same_peak(self):
        s = sizing(head_fits=True)
        s["head"]["peakGb"] = 20.62
        s["base"]["peakGb"] = 20.6
        self.assertIn("**vs base:** same peak", "\n".join(job.render_lead(s, {"grad_accum": 2})))

    def test_accumulation_is_printed_with_the_effective_batch(self):
        s = sizing(head_fits=True)
        s["head"]["batch"] = 16
        lead = "\n".join(job.render_lead(s, {"batch": 16, "grad_accum": 2}))
        self.assertIn("batch 16 x 2,048 tokens, 2 accumulation steps for an effective batch of 32", lead)

    def test_no_card_asks_for_one(self):
        body = "\n".join(job.render_lead(sizing(card=False), {}))
        self.assertIn("**Reserved card: not set.**", body)
        self.assertIn("job: {gpu: A10G, batch: 32}", body)
        self.assertIsNone(job.badge(sizing(card=False)))

    def test_failed_sizing_falls_back_to_the_server_plan(self):
        body = plan_request.render_entry(planned_entry({"ok": False, "error": "node is not on the runner"}))
        self.assertIn("could not be sized on the runner (node is not on the runner)", body)
        self.assertIn("fits a T4 at batch 32", body)

    def test_a_structural_blocker_still_shows_in_the_second_block(self):
        body = plan_request.render_entry(planned_entry(sizing(head_fits=True), legal=False))
        self.assertIn("Will it run: **no**, 1 blocker", body)
        self.assertIn("- Shape problem: linear: in 768 != 512", body)
        self.assertIn(":no_entry: blocked", body.splitlines()[0])

    def test_footer_names_the_vendored_bundle(self):
        body = plan_request.assemble_comment([planned_entry(sizing())], "0.6.0")
        self.assertIn("computed on the runner by the plan bundle vendored in neurarch-bot", body)
        self.assertNotIn(chr(0x2014), body)
        self.assertNotIn(chr(0x2013), body)


class Verdict(unittest.TestCase):
    def test_no_fit_fails_by_default_and_can_be_switched_off(self):
        entries = [planned_entry(sizing())]
        self.assertTrue(job.should_fail(entries, fail_on_warn=False, fail_on_no_fit=True))
        self.assertFalse(job.should_fail(entries, fail_on_warn=False, fail_on_no_fit=False))

    def test_a_blocker_fails_whatever_the_memory(self):
        entries = [planned_entry(sizing(head_fits=True), legal=False)]
        self.assertTrue(job.should_fail(entries, fail_on_warn=False, fail_on_no_fit=False))

    def test_fitting_and_legal_is_green(self):
        self.assertFalse(job.should_fail([planned_entry(sizing(head_fits=True))], False, True))

    def test_no_card_never_fails_on_memory(self):
        self.assertFalse(job.should_fail([planned_entry(sizing(card=False))], False, True))

    def test_accumulation_split_keeps_the_effective_batch(self):
        self.assertEqual(job.accumulation_split(32, 17), (16, 2))
        self.assertEqual(job.accumulation_split(32, 16), (16, 2))
        self.assertEqual(job.accumulation_split(32, 40), (32, 1))
        self.assertEqual(job.accumulation_split(31, 10), (10, 4))  # a prime target: the cap, rounded up


class Config(unittest.TestCase):
    def test_model_job_merges_over_the_top_level_block(self):
        config = {"job": {"gpu": "A10G", "batch": 32}, "models": []}
        self.assertEqual(job.parse_job(config, {"job": {"batch": 8, "bogus": 1}}), {"gpu": "A10G", "batch": 8})
        self.assertIsNone(job.parse_job({}, {}))

    def test_references_are_read_at_the_given_root(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "configs"))
            with open(os.path.join(d, "configs", "train.yaml"), "w") as f:
                f.write("gpu: A10G\ntrain:\n  batch_size: 24\nseq_len: 2048\n")
            spec = {"gpu": "configs/train.yaml:gpu", "batch": "configs/train.yaml:train.batch_size",
                    "seq_len": "configs/train.yaml:seq_len", "grad_accum": "configs/train.yaml:missing",
                    "epochs": "configs/nope.yaml:epochs"}
            resolved, problems = job.resolve_job(spec, d)
            self.assertEqual(resolved["gpu"], "A10G")
            self.assertEqual(resolved["batch"], 24)
            self.assertEqual(resolved["seq_len"], 2048)
            self.assertNotIn("grad_accum", resolved)
            self.assertEqual(len(problems), 2)
            self.assertEqual(job.job_files(spec), ["configs/nope.yaml", "configs/train.yaml"])

    def test_a_non_number_batch_is_a_problem_not_a_crash(self):
        resolved, problems = job.resolve_job({"batch": "lots", "seq_len": 0}, ".")
        self.assertEqual(resolved, {})
        self.assertEqual(len(problems), 2)

    def test_a_change_to_the_job_config_alone_affects_the_model(self):
        model = {"file": "models/doc_lm.py", "job_files": ["configs/pretrain.yaml"]}
        self.assertTrue(bot.is_affected(model, ["configs/pretrain.yaml"], ["models/doc_lm.py"]))
        self.assertFalse(bot.is_affected(model, ["configs/other.yaml"], ["models/doc_lm.py"]))


@unittest.skipUnless(HAS_NODE, "node is not installed")
class SizeWithTheBundle(unittest.TestCase):
    """The arithmetic is the vendored bundle's; these pin the wiring and the card table."""

    def size(self, job_spec, base_job=None, base=True):
        with tempfile.TemporaryDirectory() as d:
            out = job.size_job(FIXTURE, FIXTURE if base else None, job_spec, base_job, [], d)
        self.assertTrue(out.get("ok"), out)
        return out

    def test_peak_grows_with_batch_and_the_largest_batch_fits(self):
        small = self.size({"gpu": "T4", "batch": 8, "seq_len": 256})
        big = self.size({"gpu": "T4", "batch": 64, "seq_len": 256})
        self.assertGreater(big["head"]["peakGb"], small["head"]["peakGb"])
        self.assertEqual(small["card"], {"name": "T4 (16GB)", "gb": 16})
        mb = small["head"]["maxBatch"]
        at_max = self.size({"gpu": "T4", "batch": mb, "seq_len": 256})["head"]
        over = self.size({"gpu": "T4", "batch": mb + 1, "seq_len": 256})["head"]
        self.assertTrue(at_max["fits"])
        self.assertFalse(over["fits"])

    def test_the_base_is_sized_with_its_own_job(self):
        out = self.size({"gpu": "A10G", "batch": 64, "seq_len": 256}, base_job={"gpu": "A10G", "batch": 8, "seq_len": 256})
        self.assertEqual(out["base"]["batch"], 8)
        self.assertGreater(out["head"]["peakGb"], out["base"]["peakGb"])

    def test_an_unknown_card_is_named_not_guessed(self):
        out = self.size({"gpu": "RTX 9090", "batch": 8})
        self.assertIsNone(out["card"])
        self.assertIn("not a card the plan knows", out["cardError"])

    def test_card_names_match_the_bundle(self):
        # The card table is the one thing size_job.mjs keeps locally. Every card the bundle can
        # name as the smallest fit must be in it, or a reserved card would never match.
        script = (
            "import { CARDS, matchCard } from %s;\n"
            "const P = await import(%s);\n"
            "import fs from 'node:fs';\n"
            "const g = JSON.parse(fs.readFileSync(%s, 'utf8'));\n"
            "const seen = new Set();\n"
            "for (const b of [1, 64, 256, 512, 1024, 1500, 2000, 3000, 4000]) {\n"
            "  const p = P.buildPlan(g, { run: { batchSize: b }, seqLen: 256 });\n"
            "  if (p.cost.smallestFitGpu) seen.add(p.cost.smallestFitGpu);\n"
            "}\n"
            "const names = CARDS.map(c => c.name);\n"
            "console.log(JSON.stringify({ seen: [...seen], missing: [...seen].filter(n => !names.includes(n)),\n"
            "  a10g: matchCard('a10g')?.name, a100: matchCard('A100 80GB')?.name, full: matchCard('L40S (48GB)')?.name }));\n"
        ) % (json.dumps("file://" + os.path.join(HERE, "size_job.mjs")),
             json.dumps("file://" + os.path.join(HERE, "vendor", "modelPlan.bundle.mjs")),
             json.dumps(FIXTURE))
        r = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertGreaterEqual(len(data["seen"]), 3, data)
        self.assertEqual(data["missing"], [])
        self.assertEqual(data["a10g"], "A10G (24GB)")
        self.assertEqual(data["a100"], "A100 (80GB)")
        self.assertEqual(data["full"], "L40S (48GB)")


if __name__ == "__main__":
    unittest.main()
