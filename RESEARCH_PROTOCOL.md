# Research protocol — SafeGrip-FRC

## Primary question

Does finite-window response inversion with a friction-resolution certificate provide useful friction estimates beyond a closely matched direct recurrent regressor and published-method adaptations?

## P1 — controlled comparison

Primary paper metrics use `--protocol controlled`:

- identical locked validation/test endpoint IDs;
- identical test seeds;
- equal validation trial count;
- fixed common tuning seeds;
- common maximum epoch/batch budget for literature comparators;
- no primary physical projection;
- validation RMSE only for hyperparameter selection;
- test remains locked until final evaluation.

Architecture/preprocessing families remain model-specific so a baseline is not deliberately weakened.

## P2 — source-setting comparison

`--protocol source-faithful` preserves each registered literature-style epoch/batch/optimizer settings where recoverable. It is reported separately and is not described as compute-matched.

## Proposal tuning

FRC tuning is restricted to ordinary approximation/optimization variables:

- GRU hidden width;
- GRU layer count;
- dropout;
- learning rate;
- weight decay;
- batch size.

The friction grid, candidate horizons, certificate deltas, and theorem definition are fixed before final testing.

## Required scientific controls

1. `direct_gru_control`: same encoder depth/width and closely matched head, trained directly on friction.
2. fixed-horizon FRC variants for every declared horizon.
3. common physical-projection control reported separately.
4. conditional theorem audit.
5. unseen-trajectory split as a secondary distribution-shift experiment when the dataset contains enough independent trajectories.

## Claim discipline

The calibration radius is empirical. The deterministic recovery theorem is claimed only conditional on its residual premise. Test-set premise frequency is an evaluation statistic, not an input to prediction.
