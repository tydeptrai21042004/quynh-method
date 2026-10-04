# NPI-v3 validation note

## Final active formulation

The active proposal is **Normalized Physical Innovation v3**:

```text
reference present: y_hat = P_q(X) + c_q,ref + s_q,ref R_theta(X,q)
reference absent:  y_hat =          c_q,direct + s_q,direct R_theta(X,q)
```

All robust statistics are fitted on training data only. No RA-NPI gain, CSI affine calibration, semantic relational contrast, or feature-standardization proposal module is present in the final active path.

## Software validation

After reverting the later CSI/RA additions and removing their tests, the complete repository suite passes:

```text
170 passed, 0 failed
```

The targeted NPI/universal subset passes:

```text
24 passed, 0 failed
```

Targeted tests verify:

- small residual coordinates when a physical reference exists;
- robust direct-target coordinates when dropout removes the reference;
- exact NPI reconstruction `P + c + s z`;
- zero initialization of the innovation head;
- curvature-aware endpoint-displacement reference;
- universal model/training/collation behavior.

## Empirical status

The repository contains prior real-data evidence showing that NPI-v3 produced the main performance jump relative to the original physical-reference formulation. The final manuscript should nevertheless rerun the frozen NPI-v3 code on the declared three seeds before replacing the published aggregate table.
