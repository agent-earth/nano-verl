import tempfile
import unittest
from pathlib import Path

import torch

from verl_mini.config import load_config
from verl_mini.data import build_eval_dataset, build_train_dataset
from verl_mini.grpo import compute_group_advantages, grpo_loss
from verl_mini.report import _case_rows
from verl_mini.reward import (
    compute_accuracy,
    compute_reward,
    extract_final_integer,
    normalize_answer,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qwen_0_8b_single_gpu.yaml"


class CoreTests(unittest.TestCase):
    def test_config_and_datasets_are_deterministic_and_disjoint(self):
        cfg = load_config(CONFIG)
        train = build_train_dataset(cfg)
        eval_set = build_eval_dataset(cfg)
        self.assertEqual(train, build_train_dataset(cfg))
        self.assertEqual(len(train), 80)
        self.assertEqual(len(eval_set), 30)
        self.assertTrue({x["prompt"] for x in train}.isdisjoint(x["prompt"] for x in eval_set))

    def test_relative_data_path_is_resolved_from_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            path = root / "config.yaml"
            path.write_text("data:\n  path: data\n", encoding="utf-8")
            self.assertEqual(load_config(path).data.path, str((root / "data").resolve()))

    def test_reward_uses_final_integer_and_exact_match(self):
        self.assertEqual(extract_final_integer("work 3, answer -7"), "-7")
        self.assertEqual(extract_final_integer("#### 1,200.50"), "1200.5")
        self.assertEqual(normalize_answer("1200.500"), "1200.5")
        self.assertEqual(compute_reward("answer -7", "-7"), 1.0)
        self.assertEqual(compute_reward("answer 7", "-7"), 0.1)
        self.assertEqual(compute_accuracy(["8", "9"], ["8", "10"]), 0.5)

    def test_group_advantages_and_loss(self):
        advantages = compute_group_advantages([[0.0, 1.0], [0.5, 0.5]])
        self.assertAlmostEqual(sum(advantages[0]), 0.0, places=5)
        self.assertEqual(advantages[1], [0.0, 0.0])
        loss = grpo_loss(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, -1.0]))
        self.assertAlmostEqual(float(loss), -0.5)

    def test_unknown_config_field_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("model:\n  unknown: true\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_report_cases_cover_available_change_types(self):
        states = [(False, True), (False, False), (True, True), (True, False)]
        before = {"examples": []}
        after = {"examples": []}
        for index, (old_correct, new_correct) in enumerate(states):
            common = {
                "prompt": f"p{index}",
                "answer": str(index),
                "response": str(index),
                "extracted_answer": str(index),
            }
            before["examples"].append({**common, "correct": old_correct})
            after["examples"].append({**common, "correct": new_correct})
        rows = _case_rows(before, after, max_cases=4)
        for change in ("wrong -> correct", "wrong -> wrong", "correct -> correct", "correct -> wrong"):
            self.assertIn(change, rows)


if __name__ == "__main__":
    unittest.main()
