# DoG reading notes

Read for implementation on 29 October 2026. Primary source: Ivgi, Hinder and Carmon, [DoG is SGD's Best Friend (ICML 2023)](https://proceedings.mlr.press/v202/ivgi23a/ivgi23a.pdf). Reading focus: Sections 1–4, especially Definition 3.3, Proposition 3.6, Corollary 3.8, Theorem 3.10, and Appendix E. This is not an independent verification of every proof.

## Paper findings

SGD's learning-rate search is costly. Ordinary DoG replaces that external scale with maximum displacement divided by accumulated gradient norm (Sections 1, 3.1). A small positive initial distance prevents a zero first step.

The analysis assumes a convex objective on a closed convex domain with an attained minimum, unbiased stochastic subgradients and bounded gradient norms. Ordinary DoG's useful bounds depend on controlling iterate movement. Theorem 3.10 instead concerns T-DoG, a tamed denominator, a selected radius-weighted average, and an initial radius at most three times the distance to an optimum. These are not guarantees for our neural networks or raw last iterate.

The empirical scope spans 23 tasks and eight architectures overall, predominantly transfer learning. The main comparison uses tuned scheduled baselines, polynomial averaging and mostly five seeds. Our three small CPU tasks trained from scratch are a narrower adaptation, not a reproduction of those benchmark scores. Read Section 4.1 and Appendix E for the tuning and evaluation differences.

## Reference implementation

The [official repository](https://github.com/formll/dog) recommends polynomial averaging. Its default relative initial radius is 1e-6; numerical epsilon is 1e-8. Its `DoG` class computes one norm per parameter group, whereas `LDoG` computes separate norms per tensor. We use one group. Epsilon enters the accumulated squared gradient once.

- Inspected repository HEAD: `8d45cf569e3824212852d31cc2ae1da7ead34545` (retrieved October 29, 2026).
- Executed reference: PyPI `dog-optimizer==1.0.3`; its actual installed source SHA-256 is captured in each check's JSON. The PyPI package and repository HEAD are separately identified, not assumed byte-identical.
- [Pinned reference update implementation](https://github.com/formll/dog/blob/8d45cf569e3824212852d31cc2ae1da7ead34545/src/dog/dog.py).
- [Pinned averaging implementation](https://github.com/formll/dog/blob/8d45cf569e3824212852d31cc2ae1da7ead34545/src/dog/averager.py).

## Decisions for this project

Implement ordinary global DoG independently, without momentum, weight decay, projection, layerwise adaptation or the tamed denominator. Keep SGD independently implemented as well. The official class is a test oracle only; production code never calls it.

The current scope validates raw optimizer updates. Polynomial averaging should be integrated into the future training layer for both methods under an agreed evaluation policy; it is not implemented here. Agree final versus best-validation checkpoint selection before tuning. Do not infer from these checks that one optimizer trains better.

Questions for the next stage: Does DoG reduce tuning cost on our tasks? Does its response to scaling persist with the chosen numerical epsilon? How much does conditioning limit a global scalar step size? What sensitivity remains to the initial radius? These are experimental hypotheses, not established results.

See `implementation_spec.md` for exact executable semantics and `implementation_check.md` for measured verification.
