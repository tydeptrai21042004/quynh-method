# Validation

## SafeGrip-CI v1.4 checks

The regression suite covers data leakage controls, LiRA preparation, physics support, literature comparator contracts, historical proposal compatibility, and the active v1.4 point path.

v1.4-specific tests verify:

- counterfactual energy posterior outputs are finite and bounded;
- entropy identifiability is in [0,1];
- a flat energy landscape produces near-zero identifiability and no spurious correction;
- the final gate equals help probability × correction fraction;
- unconditional-physics ablation applies the complete candidate correction;
- multi-negative contrastive dynamics loss is differentiable;
- all 13 primary ablation semantic specifications are distinct where expected;
- v1.4 tuning-space parameters are registered.

Before packaging, run:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

Real-data claims require TRUST on LiRA. TRUST must run release-audit verification, preprocessing/physics diagnostics, the three-seed proposal/baseline benchmark, fairness controls, hierarchical statistics, all primary ablations, and scientific-health checks. Code validation does not substitute for a successful real-data experiment.
