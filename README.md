# Augmented Synthetic Control Lab

A transparent implementation of synthetic control for comparative case studies: simplex-constrained donor weights, residual augmentation, and in-space placebo inference.

```python
from model import fit
result = fit(treated_outcomes, donor_matrix, intervention=24)
print(result.weights, result.effect[24:])
```

## Why it is useful

When one city, firm, or country receives an intervention, a weighted combination of untreated units can form a more credible counterfactual than a single control. This implementation makes the weights, pre-period fit, counterfactual path, and placebo comparison directly inspectable.

## Boundaries

This is a research implementation, not a substitute for design knowledge. Convex-hull support, anticipation, spillovers, donor contamination, specification search, and weak pre-treatment fit can invalidate the causal interpretation. The residual augmentation here is deliberately compact; applied work should compare established estimators and report sensitivity.

Requires NumPy. Tests use `pytest -q`.
