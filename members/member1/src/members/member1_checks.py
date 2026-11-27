"""Small deterministic checks; no datasets or training results are produced."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
from pathlib import Path
import platform
import sys
import unittest
import uuid

import torch

from .member1_optimizers import DoG, SGD


GRADIENTS = [[3., 4.], [0., 5.], [-3., -4.], [2., -1.], [0., 0.]]


def parameter(values, dtype=torch.float64):
    return torch.nn.Parameter(torch.tensor(values, dtype=dtype))


def update(opt, p, gradient):
    p.grad = torch.tensor(gradient, dtype=p.dtype)
    opt.step()


class OptimizerChecks(unittest.TestCase):
    errors = {}

    def close(self, name, actual, expected, atol=1e-12, rtol=1e-12):
        actual = torch.as_tensor(actual).detach().double()
        expected = torch.as_tensor(expected, dtype=torch.float64)
        error = (actual - expected).abs().max().item()
        self.errors[name] = max(self.errors.get(name, 0.), error)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_sgd_hand_calculation(self):
        p = parameter([1., -2.])
        opt = SGD([p], lr=0.1)
        update(opt, p, [3., 4.])
        self.close("sgd_hand", p, [0.7, -2.4])
        update(opt, p, [-1., 2.])
        self.close("sgd_hand", p, [0.8, -2.6])

    def test_sgd_matches_pytorch(self):
        p, q = parameter([1., -2.]), parameter([1., -2.])
        own, reference = SGD([p], lr=0.03), torch.optim.SGD([q], lr=0.03)
        for g in GRADIENTS:
            update(own, p, g)
            update(reference, q, g)
            self.close("sgd_pytorch", p, q)

    def test_dog_first_two_steps_hand_calculation(self):
        p = parameter([0., 0.])
        opt = DoG([p], r_eps=1., eps=1e-8)
        eta0 = 1. / math.sqrt(25. + 1e-8)
        update(opt, p, [3., 4.])
        self.close("dog_hand", p, [-3 * eta0, -4 * eta0])
        self.assertAlmostEqual(opt.diagnostics["step_size"], eta0, places=14)
        eta1 = 1. / math.sqrt(50. + 1e-8)
        update(opt, p, [0., 5.])
        self.close("dog_hand", p, [-3 * eta0, -4 * eta0 - 5 * eta1])
        self.assertAlmostEqual(opt.diagnostics["grad_sq_sum"], 50. + 1e-8, places=13)

    def test_dog_scalar_oracle_and_radius_history(self):
        # Pure Python recurrence checks the complete trajectory, independent
        # of PyTorch reductions and state layout. Reversals exercise max history.
        x0 = [1., -2.]
        x = x0.copy()
        radius, total = 0.5, 1e-8
        p = parameter(x0)
        opt = DoG([p], r_eps=radius)
        for g in GRADIENTS * 4:
            radius = max(radius, math.dist(x, x0))
            total += sum(v * v for v in g)
            eta = radius / math.sqrt(total)
            x = [v - eta * grad for v, grad in zip(x, g)]
            update(opt, p, g)
            self.close("dog_scalar_oracle", p, x)
            self.assertAlmostEqual(opt.diagnostics["rbar"], radius, places=12)
            self.assertAlmostEqual(opt.diagnostics["step_size"], eta, places=12)

    def test_dog_matches_official(self):
        from dog import DoG as ReferenceDoG
        for dtype in (torch.float64, torch.float32):
            p, q = parameter([1., -2.], dtype), parameter([1., -2.], dtype)
            own = DoG([p], reps_rel=0.01, eps=1e-8)
            ref = ReferenceDoG([q], reps_rel=0.01, eps=1e-8, lr=1., weight_decay=0.)
            tolerance = 1e-12 if dtype == torch.float64 else 2e-6
            for g in GRADIENTS * 4:
                update(own, p, g)
                update(ref, q, g)
                self.close(f"dog_official_{dtype}", p, q, tolerance, tolerance)

    def test_global_norm_independent_of_tensor_partition(self):
        whole = parameter([1., -2.])
        a, b = parameter([1.]), parameter([-2.])
        full, split = DoG([whole], r_eps=0.5), DoG([a, b], r_eps=0.5)
        for g in GRADIENTS * 2:
            update(full, whole, g)
            a.grad, b.grad = torch.tensor(g[:1], dtype=a.dtype), torch.tensor(g[1:], dtype=b.dtype)
            split.step()
            self.close("global_partition", torch.cat([a, b]), whole)

    def test_checkpoint_roundtrip(self):
        for cls, kwargs in ((SGD, {"lr": 0.1}), (DoG, {"r_eps": 0.5})):
            for dtype in (torch.float32, torch.float64):
                p = parameter([1., -2.], dtype)
                opt = cls([p], **kwargs)
                for g in GRADIENTS[:3]:
                    update(opt, p, g)
                buffer = io.BytesIO()
                torch.save({"weights": p.detach(), "optimizer": opt.state_dict()}, buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=True)
                q = torch.nn.Parameter(saved["weights"].clone())
                restored = cls([q], **kwargs)
                restored.load_state_dict(saved["optimizer"])
                for g in GRADIENTS[3:] + GRADIENTS:
                    update(opt, p, g)
                    update(restored, q, g)
                    self.close(f"resume_{cls.__name__}_{dtype}", p, q, 0., 0.)
                    if cls is DoG:
                        self.assertEqual(opt.diagnostics, restored.diagnostics)

    def test_checkpoint_before_first_step(self):
        p, q = parameter([1., -2.]), parameter([1., -2.])
        a, b = DoG([p]), DoG([q])
        b.load_state_dict(deepcopy(a.state_dict()))
        update(a, p, [3., 4.])
        update(b, q, [3., 4.])
        self.close("resume_initial", p, q, 0., 0.)

    def test_zero_tiny_and_missing_gradients(self):
        p, unused = parameter([1., -2.]), parameter([3.])
        opt = DoG([p, unused])
        opt.step()  # no gradient: no state or initial point is captured
        self.assertEqual(opt.diagnostics["step_count"], 0)
        self.assertEqual(len(opt.state), 0)
        update(opt, p, [0., 0.])
        self.close("zero_gradient", p, [1., -2.], 0., 0.)
        self.close("missing_gradient", unused, [3.], 0., 0.)
        self.assertEqual(opt.diagnostics["grad_sq_sum"], 1e-8)
        update(opt, p, [1e-20, -1e-20])
        self.assertTrue(math.isfinite(opt.diagnostics["step_size"]))
        update(opt, p, [3., 4.])
        self.assertEqual(opt.diagnostics["step_count"], 3)
        self.assertEqual(opt.diagnostics["grad_sq_sum"], 25. + 1e-8)

    def test_reject_invalid_inputs(self):
        for invalid in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SGD([parameter([1.])], lr=invalid)
            for field in ("eps", "r_eps", "reps_rel"):
                with self.assertRaises(ValueError):
                    DoG([parameter([1.])], **{field: invalid})
        p = parameter([1.])
        with self.assertRaises(ValueError):
            DoG([{"params": []}])
        with self.assertRaises(ValueError):
            DoG([p, p])
        with self.assertRaises(ValueError):
            DoG([{"params": [p]}, {"params": [parameter([2.])]}])
        for cls, kwargs in ((SGD, {"lr": 0.1}), (DoG, {})):
            p = parameter([1.])
            opt = cls([p], **kwargs)
            for invalid in (float("nan"), float("inf")):
                p.grad = torch.tensor([invalid], dtype=p.dtype)
                with self.assertRaises(FloatingPointError):
                    opt.step()
                self.close("nonfinite_rejected_without_update", p, [1.], 0., 0.)
            p.grad = torch.sparse_coo_tensor(
                [[0]], [1.], size=(1,), dtype=p.dtype, check_invariants=True
            )
            with self.assertRaises(ValueError):
                opt.step()

    def test_autograd_and_closure(self):
        for cls, kwargs in ((SGD, {"lr": 0.1}), (DoG, {"r_eps": 0.1})):
            p = parameter([1., -2.])
            opt = cls([p], **kwargs)
            def closure():
                opt.zero_grad()
                loss = p.square().sum() / 2
                loss.backward()
                return loss
            initial_loss = closure().item()
            returned = opt.step(closure)
            self.assertEqual(returned.item(), initial_loss)
            self.assertLess(closure().item(), initial_loss)


def run_checks(notes_dir):
    """Write unique check evidence plus a latest readable summary; fail loudly."""
    torch.set_num_threads(1)
    OptimizerChecks.errors = {}
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(OptimizerChecks)
    )
    from dog import DoG as ReferenceDoG
    source = Path(inspect.getsourcefile(ReferenceDoG))
    timestamp = datetime.now(timezone.utc)
    run_id = "check_" + timestamp.strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    directory = Path(notes_dir) / "checks" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "run_id": run_id, "timestamp_utc": timestamp.isoformat(),
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "max_abs_errors": OptimizerChecks.errors,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "device": "cpu", "threads": 1,
                        "torch": torch.__version__, "numpy": importlib.metadata.version("numpy")},
        "reference": {"distribution": "dog-optimizer",
                      "version": importlib.metadata.version("dog-optimizer"),
                      "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "url": "https://github.com/formll/dog"},
        "implementation_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("member1_optimizers.py"))
        },
        "tolerances": {"float64": {"atol": 1e-12, "rtol": 1e-12},
                       "official_float32": {"atol": 2e-6, "rtol": 2e-6},
                       "checkpoint_continuation": "bitwise equality"},
    }
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / "unittest.txt").write_text(stream.getvalue())
    summary = ["# Member 1 implementation check", "",
               f"Status: **{report['status']}**. Tests: {result.testsRun}; failures: {len(result.failures)}; errors: {len(result.errors)}.",
               f"UTC: {timestamp.isoformat()}", "",
               f"Evidence: `checks/{run_id}/results.json` and `unittest.txt`.", "",
               "Independent code: `src/members/member1_optimizers.py`; checks: `src/members/member1_checks.py`.",
               "References: hand calculations, a Python scalar recurrence, PyTorch SGD, and official dog-optimizer 1.0.3 (actual version/hash in JSON).",
               "CPU only; one thread; float64 default; official comparison and resume also checked in float32.",
               "Default tolerance: atol=rtol=1e-12; official float32: 2e-6; checkpoint continuation: exact equality.", "",
               "| Comparison | Maximum absolute error |", "| --- | ---: |"]
    summary += [f"| {key} | {value:.17g} |" for key, value in sorted(OptimizerChecks.errors.items())]
    summary += ["", "Run from the repository root:", "", "```sh",
                ".venv/bin/python scripts/run_member.py --member 1 --phase check", "```", "",
                "These checks validate the update kernel only. They do not establish training quality, convergence guarantees, or readiness of smoke/tune/main/diagnostic/export phases.",
                "Iterate averaging and final neural-network evaluation policy remain future work.", ""]
    (Path(notes_dir) / "implementation_check.md").write_text("\n".join(summary))
    print(stream.getvalue())
    print(f"Evidence: {directory}")
    if not result.wasSuccessful():
        raise RuntimeError("Member 1 correctness checks failed; inspect saved evidence")
    return report
