"""Every `run:` block in action.yml has to parse as bash.

Written after breaking it: an unbalanced quote in the neurarch-mcp install step
shipped in v0.5.1 and took down the whole Action on the next pull request with
`unexpected EOF while looking for matching '"'`, after which every later step
was skipped. A composite action's shell blocks are strings inside YAML, so
neither the YAML parser nor any Python test looked at them, and the first
reader was a customer's CI.
"""
import os
import subprocess
import unittest

import yaml

ACTION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "action.yml")


class ActionYaml(unittest.TestCase):
    def setUp(self):
        with open(ACTION) as fh:
            self.doc = yaml.safe_load(fh)

    def test_every_run_block_is_valid_bash(self):
        steps = self.doc["runs"]["steps"]
        checked = 0
        for i, step in enumerate(steps):
            run = step.get("run")
            if not run:
                continue
            checked += 1
            # GitHub substitutes ${{ }} before bash sees it; a bare placeholder
            # is a word, so leaving it in is close enough for a syntax check.
            proc = subprocess.run(["bash", "-n"], input=run, text=True, capture_output=True)
            self.assertEqual(
                proc.returncode, 0,
                "step %d (%s) is not valid bash:\n%s" % (i, step.get("name"), proc.stderr),
            )
        self.assertGreater(checked, 3, "expected several shell steps; did the file move?")

    def test_the_mcp_version_is_pinned(self):
        default = self.doc["inputs"]["mcp-version"]["default"]
        self.assertRegex(str(default), r"^\d+\.\d+\.\d+$")
