"""Unit tests for report.py: body assembly, the embedded block's round trip, the drift check
and the week-over-week diff. No torch, no network, no GitHub."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report  # noqa: E402
from test_plan_request import plan_fixture  # noqa: E402


def entry(target, name, params=141322, layers=21, legal=True, warnings=0, fingerprint="857d84af",
          cost=0.043, fits="A10G (24GB)", in_shape=None, out_shape=None):
    fixture = plan_fixture(name, legal=legal)
    plan = fixture["plan"]
    plan["model"].update({"name": name, "layers": layers, "params": params, "fingerprint": fingerprint,
                          "inputShape": in_shape or [3, 32, 32], "outputShape": out_shape or [10]})
    plan["cost"].update({"estCostUsd": cost, "fitsGpu": fits})
    plan["run"]["warnings"] = [{"title": "Attention without positional encoding",
                               "detail": "4 attention layer(s) present but no positional encoding found.",
                               "fix": "Add a PositionalEncoding layer."}] * warnings
    plan.pop("diff", None)
    return {"target": target, "input": "1,3,32,32", "status": "planned", "plan": plan,
            "markdown": fixture["markdown"], "legal": legal,
            "blockers": len(plan["run"]["blockers"]), "warnings": warnings}


def two_models():
    return [
        entry("models/small_cnn.py:SmallCNN", "SmallCNN", params=141322, layers=21),
        entry("models/tiny_gpt.py:TinyGPT", "TinyGPT", params=10700000, layers=42, warnings=1,
              fingerprint="1a2b3c4d", cost=0.61, fits="A100 40GB", in_shape=[32], out_shape=[32, 1000]),
    ]


class BodyAssemblyTests(unittest.TestCase):
    def test_two_models_in_one_table(self):
        body = report.assemble_report(two_models(), version="0.2.0", commit="abc1234def",
                                      repo="neurarch-ai/model-ci-example", generated="2026-09-03T13:00:00Z")
        lines = body.split("\n")
        self.assertEqual(lines[0], report.MARKER)
        self.assertEqual(lines[1], "## Neurarch weekly model report")
        self.assertIn("| Model | Layers | Params | Shape | Will it run | GPU fit | Est. cost | Fingerprint |", body)
        self.assertIn("| `models/small_cnn.py:SmallCNN` | 21 | 141.3K | `[3,32,32]` -> `[10]` "
                      "| :white_check_mark: yes | A10G (24GB) | ~$0.04 | `857d84af` |", body)
        self.assertIn("| `models/tiny_gpt.py:TinyGPT` | 42 | 10.7M | `[32]` -> `[32,1000]` "
                      "| :warning: yes (1 warning) | A100 40GB | ~$0.61 | `1a2b3c4d` |", body)
        # Every section is present, in order.
        for a, b in (("### Blockers and warnings", "### Changed since the last report"),
                     ("### Changed since the last report", "### Documentation drift")):
            self.assertLess(body.index(a), body.index(b), "%s must come before %s" % (a, b))
        self.assertIn("record, not a gate", body)
        self.assertIn("neurarch-bot v0.2.0", body)
        self.assertIn("[`abc1234`](https://github.com/neurarch-ai/model-ci-example/commit/abc1234def)", body)

    def test_findings_section_quotes_every_blocker_and_warning(self):
        entries = two_models()
        entries[0] = entry("models/bad.py:Bad", "Bad", legal=False)
        body = report.assemble_report(entries, version="0.2.0")
        self.assertIn("1 blocker and 1 warning across 2 models.", body)
        self.assertIn("#### `models/bad.py:Bad`", body)
        self.assertIn("embed dim (384) must be divisible by numHeads (5)", body)
        self.assertIn("Attention without positional encoding", body)
        self.assertIn(":no_entry: no (1 blocker)", body)

    def test_clean_report_says_so(self):
        body = report.assemble_report([two_models()[0]], version="0.2.0")
        self.assertIn("None. Every model in the config traces", body)
        self.assertIn("No parameter count written next to a model disagrees", body)

    def test_untraceable_model_keeps_its_row(self):
        entries = [{"target": "models/broken.py:Broken", "input": "1,3,32,32", "status": "trace_failed",
                    "error": "Traceback\nImportError: no module named torchvision",
                    "legal": True, "blockers": 0, "warnings": 0}]
        body = report.assemble_report(entries, version="0.2.0")
        self.assertIn("| `models/broken.py:Broken` | - | - | - | :no_entry: could not trace | - | - | - |", body)
        self.assertIn("ImportError: no module named torchvision", body)


class DataBlockRoundTripTests(unittest.TestCase):
    def test_round_trip(self):
        entries = two_models()
        body = report.assemble_report(entries, version="0.2.0", commit="abc1234def",
                                      generated="2026-09-03T13:00:00Z")
        self.assertTrue(body.rstrip().endswith(report.DATA_CLOSE))
        data = report.parse_report_data(body)
        self.assertEqual(data["version"], report.DATA_VERSION)
        self.assertEqual(data["generated"], "2026-09-03T13:00:00Z")
        self.assertEqual(data["commit"], "abc1234def")
        self.assertEqual([m["target"] for m in data["models"]],
                         ["models/small_cnn.py:SmallCNN", "models/tiny_gpt.py:TinyGPT"])
        cnn = data["models"][0]
        self.assertEqual(cnn["params"], 141322)
        self.assertEqual(cnn["layers"], 21)
        self.assertEqual(cnn["fingerprint"], "857d84af")
        self.assertEqual(cnn["outputShape"], [10])
        self.assertIs(cnn["legal"], True)
        # A report parsed back out of its own body diffs against itself as unchanged.
        self.assertEqual(report.diff_reports(data, [report.model_row(e) for e in entries]), [])

    def test_no_raw_comment_terminator_escapes_the_block(self):
        entries = [entry("models/a.py:A-->x", "A")]
        body = report.assemble_report(entries, version="0.2.0")
        block = body[body.index(report.DATA_OPEN) + len(report.DATA_OPEN):]
        self.assertNotIn("-->", block[:block.index(report.DATA_CLOSE)])
        self.assertEqual(report.parse_report_data(body)["models"][0]["target"], "models/a.py:A-->x")

    def test_missing_or_broken_block(self):
        self.assertIsNone(report.parse_report_data(None))
        self.assertIsNone(report.parse_report_data("## Some issue somebody opened by hand"))
        self.assertIsNone(report.parse_report_data(report.DATA_OPEN + "{not json" + report.DATA_CLOSE))
        self.assertIsNone(report.parse_report_data(report.DATA_OPEN + '{"models":3}' + report.DATA_CLOSE))


class ChangedSinceLastReportTests(unittest.TestCase):
    def previous(self, **overrides):
        rows = [report.model_row(e) for e in two_models()]
        rows[0].update(overrides)
        return {"version": 1, "generated": "2026-08-27T13:00:00Z", "commit": "old1234", "models": rows}

    def test_first_report_has_nothing_to_compare(self):
        body = report.assemble_report(two_models(), version="0.2.0")
        self.assertIn("This is the first report, so there is nothing to compare against.", body)

    def test_nothing_changed(self):
        body = report.assemble_report(two_models(), version="0.2.0", previous=self.previous())
        self.assertIn("Nothing changed since 2026-08-27", body)

    def test_params_layers_and_fingerprint_move(self):
        previous = self.previous(params=971900, layers=25, fingerprint="deadbeef")
        body = report.assemble_report(two_models(), version="0.2.0", previous=previous)
        self.assertIn("Against the report of 2026-08-27:", body)
        self.assertIn("- `models/small_cnn.py:SmallCNN`: params 971.9K -> 141.3K (-85%); "
                      "layers 25 -> 21; fingerprint `deadbeef` -> `857d84af`", body)
        self.assertNotIn("`models/tiny_gpt.py:TinyGPT`:", body[body.index("### Changed since"):])

    def test_new_gone_and_legality(self):
        previous = self.previous()
        previous["models"].append(report.model_row(entry("models/gone.py:Gone", "Gone")))
        current = two_models()
        current.append(entry("models/fresh.py:Fresh", "Fresh", params=2100000))
        current[1] = entry("models/tiny_gpt.py:TinyGPT", "TinyGPT", params=10700000, layers=42,
                           legal=False, fingerprint="1a2b3c4d", cost=0.61, fits="A100 40GB",
                           in_shape=[32], out_shape=[32, 1000])
        lines = report.diff_reports(previous, [report.model_row(e) for e in current])
        joined = "\n".join(lines)
        self.assertIn("`models/fresh.py:Fresh`: **new in this report** (2.1M params)", joined)
        self.assertIn("`models/gone.py:Gone`: **no longer listed** in the config", joined)
        self.assertIn("**will no longer run** (1 blocker(s))", joined)
        self.assertIn("warnings 1 -> 0", joined)

    def test_cost_moves_only_when_it_is_visible(self):
        rows = [report.model_row(e) for e in two_models()]
        previous = {"generated": "2026-08-27T13:00:00Z",
                    "models": [dict(rows[0], estCostUsd=0.0431), dict(rows[1], estCostUsd=1.20)]}
        lines = "\n".join(report.diff_reports(previous, rows))
        self.assertNotIn("SmallCNN", lines)          # $0.0431 -> $0.043 is below half a cent
        self.assertIn("est. cost $1.20 -> $0.61", lines)

    def test_short_comment_names_what_changed(self):
        entries = two_models()
        note = report.change_comment(self.previous(params=971900), entries, drift=[], commit="abc1234def")
        self.assertIn("Report refreshed at `abc1234`.", note)
        self.assertIn("Changed since 2026-08-27", note)
        self.assertIn("params 971.9K -> 141.3K", note)
        quiet = report.change_comment(self.previous(), entries, drift=[])
        self.assertIn("Nothing changed since 2026-08-27.", quiet)
        first = report.change_comment(None, entries, drift=[])
        self.assertIn("First report", first)
        self.assertIn("2 models traced", first)


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name
        os.makedirs(os.path.join(self.root, "models"))
        self.addCleanup(self.dir.cleanup)
        self.cnn = {"target": "models/small_cnn.py:SmallCNN", "file": "models/small_cnn.py"}
        self.gpt = {"target": "models/tiny_gpt.py:TinyGPT", "file": "models/tiny_gpt.py"}

    def write(self, name, text):
        with open(os.path.join(self.root, "models", name), "w", encoding="utf-8") as f:
            f.write(text)

    def test_fires_when_the_card_disagrees(self):
        self.write("MODEL_CARD.md", "# SmallCNN\n\n| Parameters | 1,200,000 |\n")
        found = report.doc_drift(self.cnn, 141322, self.root, other_models=[self.gpt])
        self.assertIsNotNone(found)
        self.assertEqual(found["doc"], "models/MODEL_CARD.md")
        self.assertEqual(found["claimed"], 1200000.0)
        self.assertEqual(found["traced"], 141322.0)
        self.assertGreater(found["offBy"], 700)
        rendered = report.render_drift([found])
        self.assertIn("`models/MODEL_CARD.md` says 1.2M", rendered)
        self.assertIn("the trace says 141.3K", rendered)

    def test_silent_when_the_card_agrees(self):
        self.write("MODEL_CARD.md", "# SmallCNN\n\nAbout 141.3K params in total.\n")
        self.assertIsNone(report.doc_drift(self.cnn, 141322, self.root, other_models=[self.gpt]))
        self.write("MODEL_CARD.md", "# SmallCNN\n\nParameters: 141,322\n")
        self.assertIsNone(report.doc_drift(self.cnn, 141322, self.root, other_models=[self.gpt]))

    def test_silent_when_no_number_is_written(self):
        self.write("MODEL_CARD.md", "# SmallCNN\n\nA CIFAR-10 CNN with BatchNorm. Trained on 8 GPUs.\n")
        self.assertIsNone(report.doc_drift(self.cnn, 141322, self.root, other_models=[self.gpt]))

    def test_silent_when_there_is_no_doc_at_all(self):
        self.assertIsNone(report.doc_drift(self.cnn, 141322, self.root, other_models=[self.gpt]))

    def test_a_card_about_another_model_is_not_read(self):
        """The CNN's card sits in the folder the GPT shares; it must not accuse the GPT."""
        self.write("MODEL_CARD.md", "# SmallCNN\n\n| Parameters | 1,200,000 |\n")
        self.assertIsNone(report.doc_drift(self.gpt, 10700000, self.root, other_models=[self.cnn]))

    def test_a_shared_readme_naming_both_is_read_for_both(self):
        self.write("README.md", "SmallCNN has 141,322 parameters. TinyGPT has 9,000,000 parameters.\n")
        self.assertIsNone(report.doc_drift(self.cnn, 141322, self.root, other_models=[self.gpt]))
        found = report.doc_drift(self.gpt, 10700000, self.root, other_models=[self.cnn])
        self.assertIsNotNone(found)
        self.assertEqual(found["claimed"], 9000000.0)

    def test_model_card_beats_readme(self):
        self.write("MODEL_CARD.md", "# SmallCNN\n\n141,322 params\n")
        self.write("README.md", "SmallCNN: 5M parameters\n")
        self.assertIsNone(report.doc_drift(self.cnn, 141322, self.root, other_models=[self.gpt]))

    def test_one_percent_is_the_line(self):
        self.write("MODEL_CARD.md", "# SmallCNN\n\nParameters: 142,000\n")   # +0.48%
        self.assertIsNone(report.doc_drift(self.cnn, 141322, self.root, other_models=[self.gpt]))
        self.write("MODEL_CARD.md", "# SmallCNN\n\nParameters: 145,000\n")   # +2.6%
        self.assertIsNotNone(report.doc_drift(self.cnn, 141322, self.root, other_models=[self.gpt]))

    def test_untraced_model_is_never_accused(self):
        self.write("MODEL_CARD.md", "# SmallCNN\n\nParameters: 1,200,000\n")
        self.assertIsNone(report.doc_drift(self.cnn, None, self.root, other_models=[self.gpt]))

    def test_claim_shapes(self):
        claims = report.parse_param_claims(
            "Params: 1.2M\n| Parameters | 141,322 |\nabout 10.7M parameters\n"
            "parameter count = 500\nnum_heads 6\n")
        self.assertEqual([v for v, _ in claims], [1200000.0, 141322.0, 10700000.0, 500.0])

    def test_drift_section_in_the_body(self):
        found = {"target": "models/small_cnn.py:SmallCNN", "doc": "models/MODEL_CARD.md",
                 "claimed": 1200000.0, "traced": 141322.0, "offBy": 749.1, "phrase": "Parameters | 1,200,000"}
        body = report.assemble_report(two_models(), version="0.2.0", drift=[found])
        self.assertIn("A parameter count in the docs disagrees", body)
        self.assertIn("(+749%)", body)
        note = report.change_comment(None, two_models(), drift=[found])
        self.assertIn("1 documentation drift finding.", note)


class FormattingTests(unittest.TestCase):
    def test_params(self):
        self.assertEqual(report.fmt_params(0), "0")
        self.assertEqual(report.fmt_params(999), "999")
        self.assertEqual(report.fmt_params(141322), "141.3K")
        self.assertEqual(report.fmt_params(10700000), "10.7M")
        self.assertEqual(report.fmt_params(7e9), "7.0B")
        self.assertEqual(report.fmt_params(None), "?")

    def test_money(self):
        self.assertEqual(report.fmt_money(0.0), "<$0.01")
        self.assertEqual(report.fmt_money(0.043), "$0.04")
        self.assertEqual(report.fmt_money(42.0), "$42")

    def test_data_block_is_valid_json_after_escaping(self):
        raw = report.embed_data([{"target": "a>b"}], commit="c", generated="g")
        inner = raw[len(report.DATA_OPEN):-len(report.DATA_CLOSE)]
        self.assertEqual(json.loads(inner)["models"][0]["target"], "a>b")


if __name__ == "__main__":
    unittest.main()
