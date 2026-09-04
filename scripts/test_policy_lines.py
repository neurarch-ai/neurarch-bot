"""Unit tests for policy_lines.py: which line of `.neurarch.yml` a policy finding came from.

The findings are copied from real `/api/v1/plan` responses, including the two the endpoint
reports without a `detail` (an unreadable value and an unknown GPU), because those are the
ones a reader most needs pointed at a line.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policy_lines  # noqa: E402

CONFIG = """# Models neurarch-bot plans on every pull request that touches them.

# House rules. These are this repository's own lines, not Neurarch's.
policy:
  forbid_types: [softmax]   # a Softmax head in front of CrossEntropyLoss is the classic one
  max_cost_usd: 50

models:
  - path: models/tiny_gpt.py:TinyGPT
    input: "1,32:long"
  - path: models/small_cnn.py:SmallCNN
    input: 1,3,32,32
    policy:
      max_params: 500K      # this classifier's own cap; a model's policy merges over the
                            # top-level block key by key, so softmax is still forbidden here
"""


def plan(*findings, **kwargs):
    warnings = kwargs.get("warnings") or []
    return {"run": {"legal": not findings, "blockers": list(findings), "warnings": warnings}}


MAX_PARAMS = {"title": "policy: max_params 500K exceeded (1.1M)", "detail": "max_params: 500K",
              "fix": "Reduce the model by 620.2K parameters or raise max_params.", "stage": "policy"}
SOFTMAX = {"title": "policy: forbidden layer type softmax present",
           "detail": "forbid_types: [softmax]", "stage": "policy"}
MUST_FIT = {"title": "policy: must_fit T4 16GB violated (needs 34.1 GB)",
            "detail": "must_fit: T4", "stage": "policy"}
UNREADABLE = {"title": 'policy: unreadable line (max_params: expected a size like 200M, got "big")',
              "stage": "policy"}
UNKNOWN_GPU = {"title": "policy: unknown GPU in must_fit (A10G)",
               "detail": "Known: T4, A100-40GB, H100-80GB.", "stage": "policy"}
SHAPE = {"title": "Shape problem: multiHeadAttention", "stage": "preflight",
         "detail": "embed dim (384) must be divisible by numHeads (5)"}


class LocateTests(unittest.TestCase):
    def test_top_level_and_per_model_blocks(self):
        found = policy_lines.locate_rules(CONFIG)
        self.assertEqual(found["top"]["forbid_types"]["line"], 5)
        self.assertEqual(found["top"]["forbid_types"]["text"], "forbid_types: [softmax]")
        self.assertEqual(found["top"]["max_cost_usd"], {"line": 6, "text": "max_cost_usd: 50"})
        self.assertEqual(found["models"]["models/small_cnn.py:SmallCNN"]["max_params"],
                         {"line": 14, "text": "max_params: 500K"})
        # The model with no policy of its own has no entry, rather than an empty one.
        self.assertNotIn("models/tiny_gpt.py:TinyGPT", found["models"])

    def test_a_models_own_key_is_not_read_as_a_policy_key(self):
        config = "models:\n  - path: a.py:A\n    input: 1,3,32,32\n    name: max_params\n"
        self.assertEqual(policy_lines.locate_rules(config), {"top": {}, "models": {}})

    def test_a_block_sequence_inside_a_policy_is_not_a_new_model(self):
        config = ("models:\n"
                  "  - path: a.py:A\n"
                  "    input: 1,3,32,32\n"
                  "    policy:\n"
                  "      forbid_types:\n"
                  "        - softmax\n"
                  "        - sigmoid\n"
                  "      max_layers: 400\n"
                  "  - path: b.py:B\n"
                  "    input: 1,3,32,32\n")
        found = policy_lines.locate_rules(config)
        self.assertEqual(sorted(found["models"]["a.py:A"]), ["forbid_types", "max_layers"])
        self.assertEqual(found["models"]["a.py:A"]["max_layers"]["line"], 8)
        self.assertNotIn("b.py:B", found["models"])

    def test_a_key_after_the_nested_block_is_not_a_rule(self):
        config = ("models:\n"
                  "  - path: a.py:A\n"
                  "    policy:\n"
                  "      max_params: 500K\n"
                  "    input: 1,3,32,32\n"
                  "    depth: 2\n")
        found = policy_lines.locate_rules(config)
        self.assertEqual(list(found["models"]["a.py:A"]), ["max_params"])

    def test_the_comment_is_left_off_the_quoted_rule(self):
        self.assertEqual(policy_lines.rule_text("max_params: 500K   # a cap"), "max_params: 500K")
        self.assertEqual(policy_lines.rule_text("forbid_types: [softmax] # no"), "forbid_types: [softmax]")
        # A `#` inside a value is part of the value, not the start of a comment.
        self.assertEqual(policy_lines.rule_text('name: "block #2"'), 'name: "block #2"')


class RuleKeyTests(unittest.TestCase):
    def test_every_shape_the_endpoint_reports(self):
        self.assertEqual(policy_lines.rule_key(MAX_PARAMS), "max_params")
        self.assertEqual(policy_lines.rule_key(SOFTMAX), "forbid_types")
        self.assertEqual(policy_lines.rule_key(MUST_FIT), "must_fit")
        self.assertEqual(policy_lines.rule_key(UNREADABLE), "max_params")
        self.assertEqual(policy_lines.rule_key(UNKNOWN_GPU), "must_fit")

    def test_a_structural_finding_is_not_a_policy_one(self):
        self.assertIsNone(policy_lines.rule_key(SHAPE))
        self.assertIsNone(policy_lines.rule_key({}))

    def test_violated_keys_are_unique_and_in_order(self):
        p = plan(MAX_PARAMS, SOFTMAX, warnings=[UNKNOWN_GPU, MAX_PARAMS])
        self.assertEqual(policy_lines.violated_keys(p), ["max_params", "forbid_types", "must_fit"])


class RuleLineTests(unittest.TestCase):
    def setUp(self):
        self.locations = policy_lines.locate_rules(CONFIG)

    def test_a_per_model_rule_names_the_models_own_block(self):
        lines = policy_lines.rule_lines(plan(MAX_PARAMS), self.locations,
                                        "models/small_cnn.py:SmallCNN")
        self.assertEqual(lines, ["- `max_params: 500K` is at `.neurarch.yml:14`, in this model's "
                                 "own `policy:` block."])

    def test_a_top_level_rule_names_the_top_level_block(self):
        lines = policy_lines.rule_lines(plan(SOFTMAX), self.locations,
                                        "models/small_cnn.py:SmallCNN")
        self.assertEqual(lines, ["- `forbid_types: [softmax]` is at `.neurarch.yml:5`, in the "
                                 "top-level `policy:` block."])

    def test_both_at_once_keep_their_own_line(self):
        lines = policy_lines.rule_lines(plan(MAX_PARAMS, SOFTMAX), self.locations,
                                        "models/small_cnn.py:SmallCNN")
        self.assertEqual(len(lines), 2)
        self.assertIn(".neurarch.yml:14", lines[0])
        self.assertIn(".neurarch.yml:5", lines[1])

    def test_a_model_without_the_override_gets_the_top_level_line(self):
        # TinyGPT has no policy of its own, so a cost warning is the top-level line's doing.
        lines = policy_lines.rule_lines(plan(warnings=[{"title": "policy: max_cost_usd 50 exceeded ($112.40)",
                                                        "detail": "max_cost_usd: 50", "stage": "policy"}]),
                                        self.locations, "models/tiny_gpt.py:TinyGPT")
        self.assertEqual(lines, ["- `max_cost_usd: 50` is at `.neurarch.yml:6`, in the top-level "
                                 "`policy:` block."])

    def test_the_line_links_into_the_file_when_the_commit_is_known(self):
        lines = policy_lines.rule_lines(
            plan(MAX_PARAMS), self.locations, "models/small_cnn.py:SmallCNN",
            blob_url="https://github.com/o/r/blob/abc123")
        self.assertIn("[`.neurarch.yml:14`](https://github.com/o/r/blob/abc123/.neurarch.yml#L14)",
                      lines[0])

    def test_a_config_in_a_subdirectory_keeps_its_path(self):
        lines = policy_lines.rule_lines(plan(SOFTMAX), self.locations, "x", "ci/neurarch.yml",
                                        "https://github.com/o/r/blob/abc123")
        self.assertIn("ci/neurarch.yml:5", lines[0])
        self.assertIn("/ci/neurarch.yml#L5", lines[0])

    def test_a_rule_no_line_carries_says_so_rather_than_guessing(self):
        lines = policy_lines.rule_lines(plan(MUST_FIT), self.locations, "models/small_cnn.py:SmallCNN")
        self.assertEqual(len(lines), 1)
        self.assertIn("no line of `.neurarch.yml` spells it out", lines[0])
        self.assertNotIn(".neurarch.yml:", lines[0])

    def test_a_structural_blocker_produces_no_section(self):
        self.assertEqual(policy_lines.rule_lines(plan(SHAPE), self.locations, "x"), [])
        self.assertEqual(policy_lines.render_section([]), "")

    def test_the_section_says_whose_decision_it_is(self):
        section = policy_lines.render_section(
            policy_lines.rule_lines(plan(MAX_PARAMS), self.locations, "models/small_cnn.py:SmallCNN"))
        self.assertTrue(section.startswith("**The rules that fired, and where they live**"))
        self.assertIn("decision someone here made, not a Neurarch default", section)


class CommentTests(unittest.TestCase):
    def test_the_plan_comment_carries_the_section_under_the_findings(self):
        import plan_request  # noqa: PLC0415
        from test_plan_request import plan_fixture  # noqa: PLC0415
        fixture = plan_fixture("SmallCNN", legal=False)
        entry = {"target": "models/small_cnn.py:SmallCNN", "input": "1,3,32,32", "status": "planned",
                 "plan": fixture["plan"], "markdown": fixture["markdown"], "legal": False,
                 "blockers": 1, "warnings": 0,
                 "rule_lines": ["- `max_params: 500K` is at `.neurarch.yml:14`, in this model's "
                                "own `policy:` block."]}
        body = plan_request.assemble_comment([entry], version="0.5.0")
        self.assertIn("**The rules that fired, and where they live**", body)
        self.assertIn(".neurarch.yml:14", body)
        self.assertLess(body.index("**Blockers**") if "**Blockers**" in body else 0,
                        body.index("**The rules that fired"))

    def test_an_entry_without_policy_findings_is_unchanged(self):
        import plan_request  # noqa: PLC0415
        from test_plan_request import plan_fixture  # noqa: PLC0415
        fixture = plan_fixture("SmallCNN")
        entry = {"target": "m.py:M", "input": "1,3,32,32", "status": "planned",
                 "plan": fixture["plan"], "markdown": fixture["markdown"], "legal": True,
                 "blockers": 0, "warnings": 0}
        self.assertNotIn("The rules that fired", plan_request.assemble_comment([entry]))


if __name__ == "__main__":
    unittest.main()
