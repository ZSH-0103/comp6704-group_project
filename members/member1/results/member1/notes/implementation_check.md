# Member 1 implementation check

Status: **passed**. Tests: 11; failures: 0; errors: 0.
UTC: 2026-10-29T08:43:10.762193+00:00

Evidence: `checks/check_20261029T084310Z_16fef9ac/results.json` and `unittest.txt`.

Independent code: `src/members/member1_optimizers.py`; checks: `src/members/member1_checks.py`.
References: hand calculations, a Python scalar recurrence, PyTorch SGD, and official dog-optimizer 1.0.3 (actual version/hash in JSON).
CPU only; one thread; float64 default; official comparison and resume also checked in float32.
Default tolerance: atol=rtol=1e-12; official float32: 2e-6; checkpoint continuation: exact equality.

| Comparison | Maximum absolute error |
| --- | ---: |
| dog_hand | 0 |
| dog_official_torch.float32 | 0 |
| dog_official_torch.float64 | 0 |
| dog_scalar_oracle | 4.4408920985006262e-16 |
| global_partition | 0 |
| missing_gradient | 0 |
| nonfinite_rejected_without_update | 0 |
| resume_DoG_torch.float32 | 0 |
| resume_DoG_torch.float64 | 0 |
| resume_SGD_torch.float32 | 0 |
| resume_SGD_torch.float64 | 0 |
| resume_initial | 0 |
| sgd_hand | 1.1102230246251565e-16 |
| sgd_pytorch | 0 |
| zero_gradient | 0 |

Run from the repository root:

```sh
.venv/bin/python scripts/run_member.py --member 1 --phase check
```

These checks validate the update kernel only. They do not establish training quality, convergence guarantees, or readiness of smoke/tune/main/diagnostic/export phases.
Iterate averaging and final neural-network evaluation policy remain future work.
