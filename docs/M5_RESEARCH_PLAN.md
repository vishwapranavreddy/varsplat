# M5 research plan: registration-aware temporal loop closure

26 November measured prioritization: M0 profiler job 6835 used 487.425 s in loop
closure, including 409.156 s in the pairwise wrapper; matching itself used 0.592 s.
The wrapper includes identity/GT diagnostic work, so this is not a pure Gaussian
solver timing. Favor investigating registration admission and graph preparation
over search micro-optimization. This is one diagnostic trajectory, which did not
match unprofiled seed-0 ATE; no causal speed claim follows (see RESULTS.md).

26 November 2026. Research hypotheses only. M0 -> M1 -> M2 -> M3 -> M4 -> M5
execution order remains in force; this document prepares the main research focus
without claiming completion or skipping the baseline gate.

## Analysis against recovered code

The recovered implementation is identified in MAMBA_RECOVERY.md. Its active
reserved-sensor route differs from original VarSplat Gaussian registration.

| Sub-step | Actual behavior / cost | Failure hypothesis to test |
|---|---|---|
| Observation selection | At most nine uniformly spaced keyframes; six required for disjoint fitting/admission/graph roles (`view_reservation.py`) | Small submaps are unusable; uniformly spaced views may have insufficient common visibility. Count these losses before changing selection. |
| Encoding | Independent 224x224 RGB MambaVision-T -> normalized 640D vector | No depth, rendered V, per-splat sigma2 or temporal state enters descriptor. A similar-looking view need not be registerable. |
| Retrieval | Exact CPU FAISS inner product; top five eligible unique submaps; index rebuilt on add | Search is exact but budgets and visual ranking may miss useful geometry. Increasing K may only increase backend cost. |
| Candidate retention | Candidate pairs accumulate; cached registration results are reused (`live_loop.py`) | Distinguish repeat-cache cost from fresh fitting. Track map revisions and invalidation; do not retrain from duplicate outcomes as independent examples. |
| Fitting | Feature RGB-D geometry, correction consensus and robust refinement | Low texture, bad depth, weak baseline or initialization can defeat true-loop registration. Gaussian-space refinement is not active in this route. |
| Independent admission | Reserved bidirectional depth observations | Low valid depth can be confused with evidence against the loop. Distinguish unsupported from contradictory observations. |
| Graph proposals | Unselected admission plus uniform/degree-normalized retention intersection, then final solve | Conservative repeated gates can suppress useful loops and add work. This is a hypothesis until rejection reasons and geometry are audited. |
| Information | Fixed odometry 3 cm/3 degrees; loops 2 cm/2 degrees scaled by .25 and topology | Does not represent directional observability or registration-derived calibrated covariance (`reserved_backend.py::_solve_graph`). |
| Global validation | Held-out graph views; whole proposal accepted/rejected | One weakly supported constraint can affect a proposal containing useful edges. Evaluate attribution before relaxing guards. |

The eight-scene seed-1 historical bundle had 26.35% higher mean ATE and 41.33%
less core time than its matched public baseline. Candidate corrections occurred
in only two scenes. This does not establish that rejected loops caused the ATE
loss, nor isolate Mamba from other changes. The newer desk pair had lower ATE;
scene-dependent outcomes require the full regression matrix.

## Ideas table

All expected effects below are hypotheses; no percentage gain is forecast.

| ID / proposed flag | Mechanism and hypothesis | Expected ATE / time effect | Risk and novelty status |
|---|---|---|---|
| L1 `loop.sequence_tokens` | Causal state-space aggregation of frozen RGB features, metric depth summaries, missing-depth mask, rendered V summaries and visible-splat sigma2 summaries. Train retrieval toward independently verified overlap/registration success. Keep a single-image branch for opposite-direction revisits. | Potentially fewer ambiguous candidates and more useful loops; adds token extraction/state cost, may reduce backend time. | Mamba and sequence retrieval already exist. Exact RGB-D-V/sigma2 supervision is a differentiation hypothesis, not established novelty. Compare identical tokens with mean pooling, GRU and temporal convolution. |
| L2 `loop.basin_verifier` | **Proposed original research hypothesis:** predict a registration convergence region, expected iterations and directional observability from observed/rendered RGB-D, V and responses to bounded SE3 pose probes. Choose a candidate, initialization and compute budget jointly. | Potentially recovers true loops outside the local optimizer's basin while avoiding futile long solves. Probe rendering can erase savings. | Real/render verification is prior work. I did not find this exact joint basin/cost/observability prediction in the scoped sources. Broader convergence prediction literature must be checked before claiming novelty. Include non-learned probe statistics. |
| L3 `loop.observable_information` | Use whitened RGB-D residual Jacobians for directional information; validate/inflate using reserved views and view-block resampling. Admit partially informative edges with weak directions downweighted. | Potentially improves global correction without forcing unobservable motion; extra Jacobian/calibration work. | Registration covariance and robust graph factors are established. Differentiation must come from calibrated uncertainty propagation from VarSplat evidence, not merely inverting a Hessian. |
| L4 `loop.budgeted_utility` | Rank by success probability times estimated graph benefit, divided by expected total verification/registration cost. Penalize redundant loops sharing evidence. Include a bounded exploration allocation. | Potentially maintains correction quality with fewer expensive solves. Utility/model errors may discard the best loop. | Information-gain loop selection is longstanding prior work. Novelty would require a demonstrated solver-conditioned utility model, not the score formula itself. |
| L5 `loop.deferred_evidence` | Keep unsupported candidates pending; gather later independent views or adjacent-submap evidence under a bounded time/memory budget. Reject contradictory evidence separately. | Potentially recovers small-submap/missing-depth loops without lowering false-loop standards; adds delay and bookkeeping. | Delayed/multi-view verification is prior work. Treat as a non-learned comparator first. Never use insufficient evidence as automatic acceptance. |

Recommended research centerpiece: L2, with L1 supplying candidates and L3 turning
successful registrations into defensible graph constraints. L4/L5 are useful
controls and potential system contributions, not mandatory additions.

## Mathematical specification and label discipline

For candidate j, use a local candidate transform T0 and dimensionally scaled
perturbations delta_k in se(3). Coarse renders provide RGB, depth, opacity and V
at exp(delta_k)T0. The verifier observes residual summaries/spatial features,
valid-support masks and their change across probes. Sensor depth variance stays
in squared meters; appearance V stays in normalized RGB squared units.

Training-only bounded solver runs from these initializations provide success,
iterations and convergence-region targets. A solver returning success or a low
training residual is not sufficient: use reserved bidirectional RGB-D agreement,
cycle consistency and separated observation groups. Ambiguous labels remain
unlabeled. Record failures and mine visually similar non-overlap negatives.
Sample some candidates outside current retrieval/gates to limit selection bias.
Ground-truth poses are permitted for offline evaluation, not model inputs or
self-supervised training labels. SLAM-derived labels are noisy and correlated.

Candidate utility can start with U_j = p_j * delta_I_j / (cost_j + epsilon).
delta_I is an approximate graph-information benefit, not guaranteed ATE reduction.
Always account for descriptor, rendering probes, verifier, registration, graph,
correction and I/O costs. Compare under both equal candidate count and equal
wall-time budget. Do not cherry-pick only equal-count results.

For L3, H = J^T W J is an information approximation under its residual model.
H^-1 alone is not calibrated covariance. Pose increments must use a declared
tangent convention and units; transport information correctly into graph factors.
Check weak eigen-directions, correlation between views/splats and robust-loss
effects; use held-out residual coverage and view-block resampling to calibrate.
Shared source observations require redundancy controls, not independent-factor
assumptions. Compare against the exact current fixed-information graph.

Temporal token caches record frame IDs, timestamps, calibration and map revision.
Use image evidence/local relative motion rather than absolute corrected poses
as identity cues. Define state reset/re-encoding on submap revision; test reversed
traversals, different speeds and missing depth. These contracts remain IMU-ready,
but no IMU estimator is introduced before M8.

## Small reviewable diffs and implementation gate

This change is documentation only. No L1–L5 flag is currently enabled or claimed
implemented. After M0 and earlier modules, implement each behind the named flag
with an unchanged default route. First instrument a per-pair record of retrieval
rank, eligible/usable status, fitting/admission/graph reason, fresh/cache work,
time and map revision. The existing JSONL loop events provide some of this but
do not establish a fixed-positive recall denominator on their own.

## Benchmark table versus baseline

| Arm | ATE mean/std | Total time mean/std | Recall at fixed false-loop rate | GPU / render quality | Status |
|---|---|---|---|---|---|
| Original VarSplat | M0 pending full campaign | M0 pending | Not yet measured on common pair inventory | See RESULTS.md for completed runs | Reference |
| Recovered Mamba bundle | Historical values above, separate protocol | Historical values above | Not established for current comparison | Rendering not inferred from trajectory-only reports | Must reproduce |
| L1–L5 individually | Not measured | Not measured | Not measured | Not measured | Hypotheses only |

## Ablations and promotion

1. Construct one fixed offline candidate/positive inventory to compare original
   gated NetVLAD, ungated NetVLAD, current RGB MambaVision and DINOv2. Report
   recall@K, precision-recall and all denominators. Do not compare recall across
   different submap-positive populations without also reporting a common set.
2. L1: RGB -> RGB+depth -> +V -> +sigma2; independent images -> pooling -> GRU/TCN
   -> Mamba with identical features and training split. Report warm/cold latency,
   state size, parameter count, checkpoint size and training GPU-hours.
3. L2: current checks -> analytic probe score -> single-pose learned verifier ->
   pose-probe verifier -> basin/cost multitask objective. V and depth ablations;
   compare gain against the cost of spending those probes on optimization itself.
4. L3: fixed information -> Hessian approximation -> held-out calibrated factors
   -> correlation-aware admission. Report rejection rates and covariance coverage.
5. L4/L5: fixed top-K vs utility budget; immediate rejection vs deferred evidence.
   Measure detection-to-correction delay, false retained loops and useful loops lost.
6. Run complete SLAM on >=3 paired seeds, then untouched scenes. Report SE3 ATE,
   tracking/mapping/loop and full wall times, GPU memory, fixed-frame PSNR/MS-SSIM/
   LPIPS, failure counts and every ablation. Promote only measured wins; stop
   iterating when gains are below approximately 2% and measurement uncertainty is
   small enough to support that conclusion. Keep Mamba only if recall and runtime win.

## Novelty check and open questions

Already established: [LoopSplat](https://arxiv.org/abs/2408.10154) Gaussian
registration; [MMG-SLAM](https://github.com/CoolSunflower/MMG-SLAM) MambaVision
descriptors; [OverlapMamba](https://arxiv.org/abs/2405.07966) state-space place
recognition; [Flash-Mono](https://arxiv.org/abs/2604.03092) recurrent hidden-state
submap descriptors. [SCOUT-SLAM](https://arxiv.org/abs/2609.14634),
[UncLe-SLAM](https://arxiv.org/abs/2306.11048), [CG-SLAM](https://arxiv.org/abs/2403.16095)
and BayesianGS-SLAM constrain uncertainty/fusion claims (see LITERATURE_CHECK.md).

Closer loop-verification leads: [GSSeq](https://doi.org/10.3390/drones10090643)
already studies rendered-reference sequential verification;
[GLC-SLAM's publisher abstract](https://www.sciencedirect.com/science/article/pii/S0957417426024942)
describes uncertainty-guided descriptors and rendered-depth validation (full text
was inaccessible in this check). [TBV Radar SLAM](https://arxiv.org/abs/2301.04397)
combines retrieval, registration quality and verification.
[Information-Based Compact Pose SLAM](https://www.iri.upc.edu/files/scidoc/1096-Information-based-compact-Pose-SLAM.pdf)
already selects informative loop constraints.
[Through the Lens of Doubt](https://arxiv.org/abs/2510.13464) provides training-free
retrieval-confidence baselines. This is a scoped screen, not exhaustive proof.

Pending user decision: training/development/final-test scene split and access to
untouched test data. Replica/TUM already inspected in prior chats cannot be called
untouched final tests. No request to change the agreed ATE/runtime objectives or
two-job/storage limits is needed. No unmeasured innovation is promoted.
