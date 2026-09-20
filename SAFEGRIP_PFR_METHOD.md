# SafeGrip-PFR: Physics-Feasible Residual Estimation

## 1. Design principle

SafeGrip-PFR is deliberately minimal.  The proposal contains one learned model
and one deterministic projection.  It does not use a learned response model,
friction-grid inversion, a trust radius, an adaptive horizon, or a learned
selector.

The method is built around a decomposition of friction into a known mechanics
term and an unknown residual.

Let

\[
L_0(X)
\]

be the mechanics-derived lower-grip signal computed from the observed vehicle
state, and let \(\mu\) be the reference friction.

Define the signed residual target

\[
\boxed{r^\star=\mu-L_0(X).}
\]

A GRU \(f_\theta\) is trained on the training split to estimate this residual:

\[
\widehat r=f_\theta(X).
\]

The raw friction estimator is

\[
\boxed{\widetilde\mu=L_0(X)+f_\theta(X).}
\]

The residual is intentionally signed.  No ReLU or physical clipping is used in
training, so the neural estimator can compensate for finite mechanics-model
error without using calibration labels.

---

## 2. Exact loss equivalence

Because

\[
r^\star=\mu-L_0,
\]

we have pointwise

\[
\begin{aligned}
\big(f_\theta(X)-r^\star\big)^2
&=\big(f_\theta(X)-[\mu-L_0]\big)^2\\
&=\big(L_0+f_\theta(X)-\mu\big)^2\\
&=\big(\widetilde\mu-\mu\big)^2.
\end{aligned}
\]

Therefore residual MSE is exactly friction-space MSE under the translated
hypothesis class \(L_0+\mathcal F\).  Standardizing the residual target by a
train-only constant mean and standard deviation multiplies this objective by a
positive constant and leaves its minimizer unchanged.

No auxiliary response-energy loss is needed.

---

## 3. Calibrated physical lower endpoint

The raw mechanics model is conditional on modeling assumptions and finite
uncertainty margins.  PFR therefore calibrates only its lower endpoint.

For calibration pairs \((X_i,\mu_i)\), define

\[
s_i=L_0(X_i)-\mu_i.
\]

Let \(q_\alpha\) be the finite-sample higher empirical quantile at

\[
\frac{\lceil(n+1)(1-\alpha)\rceil}{n}.
\]

The implementation additionally clips \(q_\alpha\) below by zero so calibration
can only relax, never tighten, the mechanics lower endpoint:

\[
\boxed{L_\alpha(X)=\max\{0,L_0(X)-q_\alpha\}.}
\]

Under the standard split-conformal exchangeability assumption,

\[
\boxed{
\Pr\{\mu_{new}\ge L_\alpha(X_{new})\}\ge1-\alpha.
}
\]

PFR uses the **entire calibration split only for this scalar quantile**.  The
neural model has already been trained using training labels only.

---

## 4. Final estimator

Assume a physical upper bound

\[
\mu\le U.
\]

The feasible interval is

\[
C(X)=[L_\alpha(X),U].
\]

The final PFR estimate is the Euclidean projection of the raw residual estimate:

\[
\boxed{
\widehat\mu
=
\Pi_{C(X)}(\widetilde\mu).
}
\]

For a scalar interval this is simply

\[
\widehat\mu
=
\min\{U,\max\{L_\alpha,\widetilde\mu\}\}.
\]

Projection has no trainable parameter and requires no iterative optimization.

---

## 5. Projection theorem

### Theorem 1 — pointwise friction-error improvement

Let

\[
C=[L,U]
\]

be a closed interval, let \(p=\Pi_C(z)\), and suppose the true value satisfies
\(\mu^\star\in C\).  Then

\[
\boxed{
|p-\mu^\star|^2
\le
|z-\mu^\star|^2
-
|z-p|^2.
}
\]

Equivalently,

\[
\boxed{
|z-\mu^\star|^2-|p-\mu^\star|^2
\ge
\operatorname{dist}(z,C)^2.
}
\]

### Proof

Projection onto a closed convex set satisfies

\[
\langle z-p,y-p\rangle\le0
\qquad\text{for every }y\in C.
\]

Set \(y=\mu^\star\).  Expanding the squared distance gives

\[
|z-\mu^\star|^2
=
|z-p|^2+|p-\mu^\star|^2
+2(z-p)(p-\mu^\star).
\]

The projection inequality implies the final inner-product term is nonnegative,
therefore

\[
|p-\mu^\star|^2
\le
|z-\mu^\star|^2-|z-p|^2.
\]

This proves the claim. \(\square\)

For SafeGrip-PFR, set

\[
z=\widetilde\mu,
\qquad
p=\widehat\mu,
\qquad
C=[L_\alpha,U].
\]

The theorem is directly about the target friction error; it does not depend on
an auxiliary neural-response objective.

---

## 6. Finite-sample calibrated consequence

If the upper-bound assumption \(\mu\le U\) holds and the split-conformal lower
endpoint has nominal coverage \(1-\alpha\), then

\[
\Pr\left\{
|\widehat\mu-\mu|^2
\le
|\widetilde\mu-\mu|^2
-
\operatorname{dist}(\widetilde\mu,C)^2
\right\}
\ge1-\alpha.
\]

This statement is conditional on the usual conformal exchangeability
assumption and the validity of \(U\).  The code reports empirical lower/upper
coverage separately so these assumptions are visible rather than hidden.

---

## 7. Safety corollary

On the same coverage event,

\[
(\widehat\mu-\mu)_+
\le
(\widetilde\mu-\mu)_+.
\]

Thus projection cannot increase positive friction overestimation when the true
friction lies in the feasible interval.  For every threshold \(\delta>0\),

\[
\mathbf 1\{\widehat\mu-\mu>\delta\}
\le
\mathbf 1\{\widetilde\mu-\mu>\delta\}.
\]

---

## 8. Decisive ablation

PFR intentionally needs only three rows:

1. `direct_gru_control`: same GRU trained directly on \(\mu\);
2. `pfr_residual_raw`: \(L_0+f_\theta(X)\), before projection;
3. `safegrip_pfr`: calibrated projection of the residual estimator.

The first comparison measures the contribution of residualization.  The second
measures the contribution of the theorem-backed feasible projection.

No additional ablation is required to justify hidden gates, response networks,
or trust-region hyperparameters because PFR contains none of them.

---

## 9. Leakage/fairness contract

- input scaler: fit on train only;
- residual target mean/std: fit on train only;
- neural parameters: fit on train only, early-stopped on validation;
- \(q_\alpha\): fit on calibration only;
- test mechanics lower bound: computed from test features, never test friction labels;
- final projection: deterministic and label-free at inference;
- literature models: evaluated on the same endpoint IDs.

---

## 10. Implementation map

- mathematical operators: `src/safegrip/pfr.py`;
- training and comparison protocol: `src/safegrip/pfr_benchmark.py`;
- mechanics/conformal calibration: `src/safegrip/physics.py`;
- unit theorem tests: `tests/test_pfr.py`;
- active CLI: `safegrip benchmark --proposal pfr ...`;
- Kaggle driver: `KAGGLE_PFR_SINGLE_CELL.py`.
