# SafeGrip-FRC theorem validation protocol

The mathematical result is conditional. The code therefore separates **theorem correctness** from **how often its empirical premise holds**.

## Per test endpoint and horizon

The benchmark records:

- true friction `mu_true`;
- grid estimate `mu_hat`;
- nearest candidate-grid friction;
- true nearest-grid forward residual norm;
- calibration radius `r_H`;
- separation curve `S_H(delta)`;
- finite certificate and continuous error certificate.

For every declared `delta`, the audit evaluates

\[
\text{premise}=
\{\|Y-\widehat\Phi(q(\mu^*))\|\le r_H\}
\cap
\{S_H(\delta)>2r_H\}.
\]

Whenever this premise is true, code verifies both

\[
|\widehat\mu-q(\mu^*)|<\delta
\]

and

\[
|\widehat\mu-\mu^*|\le\delta+h_\mu/2.
\]

Any violation makes `theorem_audit.json` fail and the primary benchmark raises an error.

## Important interpretation

A finite `delta_cert` alone does not prove that the unseen true residual is below the calibration radius. The paper must report separately:

1. separation-condition rate;
2. empirical residual-premise rate on the locked test set;
3. full-premise sample count;
4. theorem violations (expected to be zero up to numerical tolerance);
5. point-estimation RMSE/MAE.

This avoids turning an empirical residual quantile into an unsupported probabilistic theorem.
