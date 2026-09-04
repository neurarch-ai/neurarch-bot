"""Unit tests for init.py: the walk, the guesses, the derived numbers and both documents.

No torch, no network, no GitHub. The repository under test is written to a temporary directory
by `make_repo`, and the measurements are the ones `/api/v1/plan` returns for graphs of that
size, so `derive_policy` is exercised on numbers of the shape it will really see.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot  # noqa: E402
import init  # noqa: E402
import policy_lines  # noqa: E402

SMALL_CNN = '''"""A small CIFAR-10 CNN."""
import torch.nn as nn


def conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch))


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(conv_bn_relu(3, 32), nn.MaxPool2d(2))
        self.classifier = nn.Linear(192, num_classes)
'''

TINY_GPT = '''import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4)
        self.mlp = nn.Linear(dim, dim)


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(1000, 128)
        self.blocks = nn.Sequential(*[Block(128) for _ in range(4)])
'''

NEEDS_ARGS = '''import torch.nn as nn


class BigGPT(nn.Module):
    def __init__(self, vocab_size, n_layer=12):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, 256)
'''

NOT_A_MODEL = '''class Trainer:
    def __init__(self):
        self.epochs = 10
'''


def make_repo(files):
    root = tempfile.mkdtemp()
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return root


def measured(target, params, layers=10, t4=True, a100=True, h100=True):
    return {"target": target, "params": params, "layers": layers,
            "gpu_fits": {"T4": t4, "A100_40": a100, "H100_80": h100},
            "fits": "T4" if t4 else ("A100-40GB" if a100 else "H100-80GB")}


class WalkTests(unittest.TestCase):
    def test_it_finds_the_models_and_skips_everything_else(self):
        root = make_repo({
            "models/small_cnn.py": SMALL_CNN,
            "models/tiny_gpt.py": TINY_GPT,
            "train.py": NOT_A_MODEL,
            "tests/test_models.py": SMALL_CNN,
            "test_smoke.py": SMALL_CNN,
            ".venv/lib/site-packages/torchvision/models/resnet.py": SMALL_CNN,
            "checkpoints/best.neurarch.json": "{}",
        })
        found = init.discover(root)
        names = [c["target"] for c in found["candidates"]]
        self.assertEqual(sorted(names), ["models/small_cnn.py:SmallCNN",
                                         "models/tiny_gpt.py:Block",
                                         "models/tiny_gpt.py:TinyGPT"])
        self.assertEqual(found["graphs"], ["checkpoints/best.neurarch.json"])
        # `train.py` has a class, but not an `nn.Module` one; the walk read it and moved on.
        self.assertEqual(found["python_files"], 3)

    def test_a_block_another_class_constructs_is_marked_a_part(self):
        root = make_repo({"models/tiny_gpt.py": TINY_GPT})
        by_name = {c["name"]: c for c in init.discover(root)["candidates"]}
        self.assertEqual(by_name["Block"]["part_of"], "TinyGPT")
        self.assertIsNone(by_name["TinyGPT"]["part_of"])

    def test_a_constructor_that_needs_arguments_is_reported_not_dropped(self):
        root = make_repo({"models/big.py": NEEDS_ARGS})
        candidate = init.discover(root)["candidates"][0]
        self.assertEqual(candidate["needs_args"], ["vocab_size"])
        self.assertEqual(init.required_arguments("def __init__(self, a, b=1, *args, **kw):"), ["a"])
        self.assertEqual(init.required_arguments("def __init__(self, d: int = 2):"), [])


class GuessTests(unittest.TestCase):
    def test_an_embedding_means_token_ids(self):
        self.assertEqual(init.guess_inputs("", TINY_GPT)[0], "1,32:long")

    def test_a_conv_wins_over_a_linear_in_the_same_class(self):
        # SmallCNN's own body has only the `nn.Linear`; the conv is behind a helper function,
        # and reading the linear would guess a flat vector for a model that wants an image.
        blocks = {b["name"]: b for b in init.class_blocks(SMALL_CNN)}
        self.assertEqual(init.guess_inputs(blocks["SmallCNN"]["body"], SMALL_CNN)[0], "1,3,32,32")

    def test_a_literal_channel_count_is_used_when_there_is_one(self):
        self.assertEqual(init.guess_inputs("nn.Conv2d(1, 16, 3)")[0], "1,1,32,32")

    def test_a_bare_linear_model_gets_its_in_features(self):
        self.assertEqual(init.guess_inputs("self.fc = nn.Linear(784, 10)"), ["1,784"])

    def test_nothing_recognisable_falls_back_to_an_image(self):
        self.assertEqual(init.guess_inputs("self.x = 1"), ["1,3,32,32"])


class DerivedNumberTests(unittest.TestCase):
    def test_three_models_give_a_cap_above_the_largest_and_the_smallest_card(self):
        models = [measured("models/small_cnn.py:SmallCNN", 141322),
                  measured("models/mnist_net.py:MnistNet", 20586),
                  measured("models/tiny_gpt.py:TinyGPT", 13500000)]
        derived = init.derive_policy(models)
        self.assertEqual(derived["max_params"]["value"], "20M")
        self.assertEqual(derived["max_params"]["kind"], "measured")
        self.assertIn("TinyGPT", derived["max_params"]["because"])
        self.assertIn("13.5M", derived["max_params"]["because"])
        self.assertEqual(derived["must_fit"]["value"], "T4")
        self.assertTrue(derived["must_fit"]["live"])

    def test_the_smallest_card_is_the_smallest_every_model_fits(self):
        models = [measured("a", 100), measured("b", 2000000000, t4=False)]
        self.assertEqual(init.derive_policy(models)["must_fit"]["value"], "A100-40GB")
        self.assertIsNone(init.smallest_gpu([measured("c", 1, t4=False, a100=False, h100=False)]))

    def test_no_card_fits_every_model_so_no_line_is_written(self):
        derived = init.derive_policy([measured("a", 1, t4=False, a100=False, h100=False)])
        self.assertNotIn("must_fit", derived)

    def test_forbid_types_is_a_suggestion_and_arrives_commented_out(self):
        derived = init.derive_policy([measured("a", 1000)])
        self.assertEqual(derived["forbid_types"]["kind"], "suggestion")
        self.assertFalse(derived["forbid_types"]["live"])
        self.assertEqual(derived["forbid_types"]["value"], "[softmax, sigmoid]")

    def test_nothing_measured_means_no_numbers_at_all(self):
        derived = init.derive_policy([])
        self.assertNotIn("max_params", derived)
        self.assertNotIn("must_fit", derived)

    def test_the_cap_is_round_and_it_is_above_the_model(self):
        self.assertEqual(init.round_cap(141322), "200K")
        self.assertEqual(init.round_cap(13500000), "20M")
        self.assertEqual(init.round_cap(20586), "50K")
        self.assertEqual(init.round_cap(900000000), "2B")
        self.assertEqual(init.round_cap(1), "2")
        for params in (1, 999, 141322, 4200000, 88000000000):
            cap = init.round_cap(params)
            value = float(cap.rstrip("KMB")) * {"K": 1e3, "M": 1e6, "B": 1e9}.get(cap[-1], 1)
            self.assertGreater(value, params, "%s must be above %d" % (cap, params))


class ConfigTests(unittest.TestCase):
    def entries(self):
        return [
            {"target": "models/small_cnn.py:SmallCNN", "file": "models/small_cnn.py",
             "name": "SmallCNN", "input": "1,3,32,32", "status": "measured", "params": 141322,
             "layers": 21, "fits": "T4"},
            {"target": "models/tiny_gpt.py:TinyGPT", "file": "models/tiny_gpt.py",
             "name": "TinyGPT", "input": "1,32:long", "status": "measured", "params": 13500000,
             "layers": 42, "fits": "T4"},
            {"target": "models/big.py:BigGPT", "file": "models/big.py", "name": "BigGPT",
             "input": "1,32:long", "status": "skipped",
             "reason": "`BigGPT()` takes `vocab_size`, so it cannot be constructed with no arguments."},
        ]

    def rendered(self):
        derived = init.derive_policy([measured("models/small_cnn.py:SmallCNN", 141322),
                                      measured("models/tiny_gpt.py:TinyGPT", 13500000)])
        return init.render_config(self.entries(), derived, "0.5.0")

    def test_it_round_trips_through_the_reader_that_will_read_it(self):
        text = self.rendered()
        root = tempfile.mkdtemp()
        path = os.path.join(root, ".neurarch.yml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        config = bot.load_config(path)
        models = bot.parse_models(config)
        self.assertEqual([m["target"] for m in models],
                         ["models/small_cnn.py:SmallCNN", "models/tiny_gpt.py:TinyGPT"])
        self.assertEqual(models[1]["input"], "1,32:long")
        self.assertEqual(models[0]["file"], "models/small_cnn.py")
        # The policy it wrote is the policy every one of those models is planned against.
        self.assertEqual(bot.parse_policy(config, models[0]), {"max_params": "20M", "must_fit": "T4"})

    def test_the_rules_it_writes_can_be_found_again_by_line(self):
        found = policy_lines.locate_rules(self.rendered())
        self.assertIn("max_params", found["top"])
        self.assertIn("must_fit", found["top"])
        self.assertEqual(found["top"]["max_params"]["text"], "max_params: 20M")
        self.assertEqual(found["models"], {})

    def test_a_model_it_could_not_trace_is_commented_out_with_the_reason(self):
        text = self.rendered()
        self.assertIn("# - path: models/big.py:BigGPT", text)
        self.assertIn("cannot be constructed with no arguments", text)
        self.assertNotIn("\n  - path: models/big.py:BigGPT", text)

    def test_the_measured_numbers_are_written_next_to_the_model_they_measured(self):
        text = self.rendered()
        self.assertIn("SmallCNN: 21 layers, 141.3K params, fits a T4. Traced with input 1,3,32,32.", text)
        self.assertIn("# forbid_types: [softmax, sigmoid]", text)
        self.assertIn("\n  max_params: 20M", text)

    def test_a_repository_where_nothing_traced_still_writes_a_usable_file(self):
        entries = [dict(self.entries()[2])]
        text = init.render_config(entries, init.derive_policy([]), "0.5.0")
        self.assertIn("Nothing here traced", text)
        self.assertIn("Nothing was measured, so nothing here is set", text)
        self.assertNotIn("\n  max_params", text)


class PullRequestBodyTests(unittest.TestCase):
    def body(self):
        entries = ConfigTests().entries()
        derived = init.derive_policy([measured("models/small_cnn.py:SmallCNN", 141322),
                                      measured("models/tiny_gpt.py:TinyGPT", 13500000)])
        return init.pr_body(entries, derived, "0.5.0", run_url="https://github.com/o/r/actions/runs/1",
                            searched=12)

    def test_it_says_which_numbers_are_measured_and_which_are_suggested(self):
        body = self.body()
        self.assertTrue(body.startswith(init.MARKER))
        self.assertIn("`max_params: 20M` (on, **measured**)", body)
        self.assertIn("`must_fit: T4` (on, **measured**)", body)
        self.assertIn("`forbid_types: [softmax, sigmoid]` (commented out, "
                      "**a suggestion, not a measurement**)", body)

    def test_every_model_it_traced_is_in_the_table_with_its_numbers(self):
        body = self.body()
        self.assertIn("| `models/small_cnn.py:SmallCNN` | `1,3,32,32` | 21 | 141.3K | T4 |", body)
        self.assertIn("| `models/tiny_gpt.py:TinyGPT` | `1,32:long` | 42 | 13.5M | T4 |", body)

    def test_what_it_could_not_trace_is_named_rather_than_dropped(self):
        body = self.body()
        self.assertIn("### Found, not listed", body)
        self.assertIn("models/big.py:BigGPT", body)

    def test_it_links_the_run_and_the_annotated_example(self):
        body = self.body()
        self.assertIn("https://github.com/o/r/actions/runs/1", body)
        self.assertIn(init.EXAMPLE_URL, body)
        self.assertIn("neurarch-bot v0.5.0", body)


class BranchTests(unittest.TestCase):
    def test_the_default_branch_gets_the_plain_name(self):
        self.assertEqual(init.branch_name("main", "main"), "neurarch/add-config")

    def test_another_branch_gets_one_of_its_own(self):
        self.assertEqual(init.branch_name("no-config", "main"), "neurarch/add-config-no-config")
        self.assertNotEqual(init.branch_name("no-config", "main"), "no-config")


class ConfigAlreadyExistsTests(unittest.TestCase):
    def run_init_in(self, root, files):
        for rel, text in files.items():
            with open(os.path.join(root, rel), "w", encoding="utf-8") as f:
                f.write(text)
        output = os.path.join(root, "outputs.txt")
        os.environ["GITHUB_OUTPUT"] = output
        try:
            code = bot.run_init(root, tempfile.mkdtemp(), None,
                                os.path.join(root, ".neurarch.yml"), {}, None)
        finally:
            os.environ.pop("GITHUB_OUTPUT", None)
        with open(output, "r", encoding="utf-8") as f:
            lines = dict(line.split("=", 1) for line in f.read().splitlines() if "=" in line)
        return code, lines

    def test_a_repository_that_has_a_config_is_left_alone(self):
        root = make_repo({"models/small_cnn.py": SMALL_CNN})
        before = "models:\n  - path: models/small_cnn.py:SmallCNN\n    input: 1,3,32,32\n"
        code, outputs = self.run_init_in(root, {".neurarch.yml": before})
        self.assertEqual(code, 0)
        self.assertEqual(outputs["init-status"], "exists")
        self.assertEqual(outputs["init-pr-url"], "")
        with open(os.path.join(root, ".neurarch.yml"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), before)

    def test_a_repository_with_nothing_in_it_writes_nothing(self):
        root = make_repo({"README.md": "# nothing here"})
        code, outputs = self.run_init_in(root, {})
        self.assertEqual(code, 0)
        self.assertEqual(outputs["init-status"], "nothing-found")
        self.assertFalse(os.path.exists(os.path.join(root, ".neurarch.yml")))


if __name__ == "__main__":
    unittest.main()
