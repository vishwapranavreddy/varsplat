# M1 preparation: uncertainty calibration and fusion

Status: analysis and CPU diagnostics only, 25 November 2026. M0 multi-seed,
multi-scene reproduction is incomplete. No estimator intervention is enabled,
no learned component has been trained, and no ATE/runtime improvement is claimed.

## Analysis

The source-grounded upstream paths and native kernel identities are in
[M0_CODE_MAP.md](M0_CODE_MAP.md); the recovered Mamba route is in
[MAMBA_RECOVERY.md](MAMBA_RECOVERY.md).

1. **Parameterization:** each splat stores three log appearance variances
   `ell_ic`, initialized to -7. `sigma2_ic = exp(ell_ic)` costs O(3N) storage/work.
   This is separate from geometric covariance derived from scale and rotation.
2. **Rendering:** with `w_i = alpha_i T_i`, the black-background kernel returns
   `V_c = clip(sum_i w_i(sigma2_ic + mu_ic^2) - (sum_i w_i mu_ic)^2, 1e-6, .2)`.
   Cost follows visible splat-pixel contributions, not just the number of pixels.
   Between-splat color disagreement and unfilled opacity affect V; it is not
   purely sensor noise. Saturation can hide error magnitude.
3. **Learning:** mapping adds `1e-4 * mean(.5*r_rgb^2/(V+eps) + .5*log(V+eps))`.
   Residual squares are detached. With `use_nlldepth`, metric depth residuals
   share the three appearance-variance channels. This mixes squared meters and
   squared normalized RGB values. The original mapping variance adjoint is
   incorrect; correcting it is an implementation repair, not an original idea.
4. **Tracking:** detached weights are
   `clip(exp(-(log(V+eps)-median(log(V+eps)))/tau_track), 1/L, L)` per channel.
   The median uses only tracking-valid pixels, while the residual path also uses
   alpha/depth masks. Medians add pixel reductions; empty masks can produce NaNs.
   Scaling all V by a constant cancels exactly when eps=0. With finite epsilon,
   sensitivity remains near the variance floor; it must be measured rather than
   assuming any scalar calibration changes pose estimation.
5. **Registration:** the Gaussian route uses detached rendered variance and
   its own masks. The upstream positional-argument error changes effective tau;
   isolate that repair separately. The recovered active Mamba sensor route
   instead uses depth-feature geometry and its nominal noise model.
6. **Retrieval:** original submap reliability reduces per-splat variance to an
   opacity-weighted scalar after centering/normalizing/clamping. It loses spatial
   information. Recovered MambaVision RGB features do not consume V or sigma2.
7. **Evaluation:** Gaussian NLL, standardized residual energy and coverage assess
   calibration; ATE and wall time assess system value. NLL improvement alone is
   insufficient, particularly under median-centered weights. Correlated residuals
   are not independent calibration samples; aggregate by scene/seed as well.

## Ideas table

Expected effects are hypotheses, not measured gains. Run these only after M0.

| ID / proposed flag | Hypothesis and expected ATE/time effect | Risks | Novelty screen / ablation |
|---|---|---|---|
| U0 `uncertainty.variance_adjoint=corrected` | Correct channel/clamp derivatives allow sigma2 to learn the intended NLL; ATE effect unknown, kernel cost expected similar. | Changes the map and weights; cannot assume a mathematically correct gradient improves ATE. | Already present in the recovered candidate. Engineering repair, not novel. Stock vs corrected kernel only; separately test registration argument binding. |
| U1 `uncertainty.depth_model=analytic` | Separate `s_d(z)=a+b*z^2` in meters and normalize geometric residuals by sensor/model depth uncertainty. Expected more stable depth weighting; small elementwise cost. | Real noise depends on sensor, incidence, boundaries and missing depth. Adding appearance V to depth variance is invalid. | Established direction in UncLe-SLAM/CG-SLAM; BayesianGS-SLAM already combines sensor and representation uncertainty. Compare fixed depth weight, analytic noise and learned noise with identical RGB weights. |
| U2 `uncertainty.calibration=conditional` | Fit a positive per-channel floor/temperature model on cross-view residuals withheld from the current map update. Spatial/conditional calibration changes weights where scalar scaling cancels. Expected ATE benefit only if it improves useful residual ordering; adds a small pixelwise head. | Labels contain pose/map errors; leakage if an image fits its own map; can learn to suppress correct static evidence. | Calibration and cross-view supervision are established. Differentiate from SCOUT-SLAM's shared feature network and MonoCal's depth disagreement target. Compare scalar, analytic conditional and learned conditional models, with equal data. |
| U3 `uncertainty.calibration_objective=centered_weight` | Proposed original research hypothesis: constrain the conditional calibrator's identifiable, median-centered output and train with both held-out residual likelihood and a penalty for assigning high actual optimizer weight to inconsistent reserved RGB-D observations. Test whether this improves system behavior beyond NLL-only calibration. Expected ATE effect unknown; training adds a weight-space objective, inference reuses U2. | Non-smooth median, biased self-supervised poses, dependence on a fixed tau, and reserved-view selection can dominate. Must compare identical architecture with objective removed. | I did not find this exact VarSplat-centered-weight objective in the scoped sources reviewed. That is a differentiation hypothesis, not established novelty. Broader task-aware calibration literature review remains required before any originality claim. |

The learned candidate must report parameter count, checkpoint bytes, cold/warm
inference latency, peak memory, training GPU-hours and dataset provenance. None
is available yet. No learned model is retained without beating its analytic
alternative on unseen scenes and measured runtime/accuracy.

## Small reviewable code diffs

`tools/analysis/uncertainty_audit.py` is an offline CPU diagnostic, explicitly
invoked rather than enabled in SLAM. It computes appearance NLL/coverage/binned
residuals and tests scalar-weight sensitivity. It reproduces PyTorch's
lower-middle median convention for even sample counts, rejects empty/invalid
selected samples and avoids mixing depth units. Five CPU tests pass.

No U0–U3 intervention is implemented in `src/` yet. Existing baseline reports and
source hashes remain intact. This is preparation for the module, not completion.

Diagnostic command, from repository root, on an explicitly recorded sample NPZ:

```bash
python -m tools.analysis.uncertainty_audit --samples appearance_samples.npz \
  --tau 50 --limit 3 --output appearance_audit.json
```

Required NPZ fields: `residual` and `variance` with shape N x 3 in normalized RGB
units, and N boolean `valid`. Output contains `calibration`, `weighting`, sample
SHA-256 and a scope statement. The sample collector must record scene, frame,
map revision and whether the observation was used to fit that map; it has not
yet been run on a SLAM dataset. No synthetic test value is a benchmark result.

## Benchmark table versus baseline

| Arm | SE3 ATE mean/std | SLAM wall mean/std | GPU memory | Rendering | Status |
|---|---|---|---|---|---|
| Unmodified VarSplat | Pending >=3 comparable seeds | Pending | Seed-0 measurements in RESULTS.md | Native and fixed-frame protocols recorded | M0 in progress |
| U0 | Not measured in isolation | Not measured | Not measured | Not measured | Old bundled evidence is insufficient |
| U1 | Not measured | Not measured | Not measured | Not measured | Proposed |
| U2 | Not measured | Not measured | Not measured | Not measured | Proposed; split pending |
| U3 | Not measured | Not measured | Not measured | Not measured | Proposed; novelty check incomplete |

## Ablation sequence and acceptance

Frozen baseline -> U0 alone -> registration-call repair alone -> independent RGB
and depth uncertainty -> scalar calibration control -> analytic conditional model
-> learned model -> centered-weight objective. Only combine individually measured
wins, and then measure interactions. Repeat paired seeds; report all failures,
both accuracy and time, and fixed-frame rendering. A <2% apparent change is not
automatically a stop rule when uncertainty spans that threshold.

IMU remains deferred. Preserve explicit timestamps, frame IDs, camera units,
coordinate conventions and the measurement/pose-correction distinction. Do not
interpret appearance variance as a future IMU or pose-factor covariance.

## Open questions before training

The scene split and final-test data location remain unresolved. The user was
asked whether to use already explored Replica/TUM as development and untouched
ScanNet/ScanNet++ as final testing. No dependent training will start before that
choice is resolved. M0 reproduction and non-learned diagnostics can continue.

## Primary sources checked

- [UncLe-SLAM](https://arxiv.org/abs/2306.11048): learned self-supervised depth uncertainty.
- [CG-SLAM](https://arxiv.org/abs/2403.16095): Gaussian depth uncertainty and primitive selection.
- [SCOUT-SLAM](https://arxiv.org/html/2609.14634v1): shared uncertainty model with multi-view supervision.
- [BayesianGS-SLAM](https://arxiv.org/html/2609.24140v1): predictive RGB/depth uncertainty from sensor and opacity components.
- [MonoCal](https://arxiv.org/html/2611.19188v1): separates scale and shape calibration without GT depth.
- [LoopSplat](https://arxiv.org/abs/2408.10154), [OverlapMamba](https://arxiv.org/abs/2405.07966), [Flash-Mono](https://arxiv.org/abs/2604.03092): constrain later registration/sequence-descriptor novelty claims.
- [MMG-SLAM source](https://github.com/CoolSunflower/MMG-SLAM): MambaVision descriptors in Gaussian SLAM already exist.
