"""Check quadratic geometry, gradient, pairing and logged target semantics."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from .member1_diagnostics import problem, conditions, run_one


class QuadraticChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[2] / "configs/member1_quadratic.json").read_text())

    def test_spectrum_gradient_and_pairing(self):
        for k, a in conditions(self.cfg):
            h, x = problem(self.cfg, k, a, 1)
            ev = torch.linalg.eigvalsh(h)
            self.assertAlmostEqual(float(ev[0]), a, places=12)
            self.assertAlmostEqual(float(ev[-1]), a*k, places=10)
            self.assertAlmostEqual(float(torch.linalg.vector_norm(x)), 1., places=14)
            h2, x2 = problem(self.cfg, k, a, 1)
            torch.testing.assert_close(h, h2, atol=0, rtol=0)
            torch.testing.assert_close(x, x2, atol=0, rtol=0)
            x.requires_grad_(True)
            loss = x @ h @ x / 2
            loss.backward()
            torch.testing.assert_close(x.grad, h @ x.detach(), atol=1e-12, rtol=1e-12)
        self.assertEqual(len(conditions(self.cfg)), 5)

    def test_objective_scaling(self):
        base, x = problem(self.cfg, 10, 1., 1)
        for a in self.cfg["objective_scales"]:
            h, initial = problem(self.cfg, 10, a, 1)
            torch.testing.assert_close(h, a * base, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(x, initial, atol=0, rtol=0)

    def test_closed_form_sgd_and_target_logging(self):
        cfg = dict(self.cfg, max_updates=5)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            r = run_one(root, cfg, "test", {}, 10, 1, 1, "sgd", "diagnostic_test", .1)
            h, x = problem(cfg, 10, 1, 1)
            expected = torch.linalg.matrix_power(torch.eye(20, dtype=torch.float64) - .1*h, 5) @ x
            expected_gap = float(expected @ h @ expected / 2)
            self.assertAlmostEqual(r["final_objective_gap"], expected_gap, places=13)
            one = run_one(root, cfg, "test", {}, 1, 1, 1, "sgd", "diagnostic_test", 1.)
            self.assertEqual(one["target_step"], 1)
            self.assertEqual(one["status"], "completed")
            never = run_one(root, dict(cfg, max_updates=1), "test", {}, 100, 1, 1, "dog", "diagnostic_test")
            self.assertFalse(never["target_reached"])
            self.assertIsNone(never["target_step"])

    def test_divergence_is_retained(self):
        cfg = dict(self.cfg, max_updates=10, divergence_relative_gap=2.)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            r = run_one(root, cfg, "test", {}, 1, 1, 1, "sgd", "diagnostic_test", 3.)
            self.assertEqual(r["status"], "diverged")
            self.assertIsNone(r["final_relative_gap"])
            self.assertTrue((root / "raw" / r["run_id"] / "metrics.jsonl").exists())
            self.assertTrue((root / "raw" / r["run_id"] / "summary.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
