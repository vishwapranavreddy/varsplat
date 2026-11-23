# M0 — source-grounded code map

Audited upstream: `4a21a0aaade40b7c42d47181108e49a8a0e695f2` (2026-11-23 checkout).
Local branch: `research/m0-reproduction`, in the history-preserving
[GitHub fork](https://github.com/vishwapranavreddy/varsplat). No original Python source or scene
configuration has been edited. All M0 code lives under `tools/m0/`.

## Analysis

`run_slam.py` loads recursive YAML configuration, applies CLI overrides, initializes
random seeds, constructs `GaussianSLAM`, calls `run()`, then calls `Evaluator.run()`.
`run_evaluation.py` only reloads saved maps/poses and evaluates them. It cannot
reconstruct the runtime or seed variability of the original SLAM execution.

| Component | Actual execution path | Mathematics and work | Failure/cost observations |
|---|---|---|---|
| Dataset | `src/entities/datasets.py:15`, `:76`, `:172`, `:213` | RGB-D/pose association; TUM nearest depth/pose within 0.08 s, then 32 Hz subsampling. ScanNet expects preprocessed RGB PNG/depth TIFF/pose text. | Dataset association and preprocessing must match. `data.frame_limit` is read; top-level `frame_limit` in base configs is ineffective. No native timestamp/IMU packet interface. |
| Orchestration | `src/entities/gaussian_slam.py:240` | Sequential tracking → possible submap save/loop correction → mapping. Frames 0 and 1 use dataset GT poses. | Keep this bootstrap in the baseline; do not call it fully GT-free. Logging, diagnostic plots, file I/O and map correction contribute to online wall time. |
| Tracking initialization | `tracker.py:161`; `tracker_utils.py:63`; `visual_odometer.py` | Constant velocity `T[k-1] inv(T[k-2]) T[k-1]`, previous pose, or Open3D RGB-D odometry. High initial residual doubles iteration count; optional odometer fallback. | Existing adaptive iteration mechanism already exists. Initialization/rendering is outside the original tracker timer. The `gt` branch returns a different shape than the normal tuple. |
| Tracking optimization | `tracker.py:78` and `:161` | Transform Gaussian centers/rotations, rasterize RGB/depth/V, depth/alpha/outlier masks, Adam on camera quaternion/translation. `L = lambda sum(w_rgb |r_rgb|) + (1-lambda) sum(|r_d|)`, including configured alpha masks. | Full splat transform and rasterization every iteration; CPU pose copies/logging synchronize. V is detached; variance weights apply to RGB, not depth. Empty valid masks can yield non-finite medians. |
| Seeding/growth | `mapper.py:77`, `:102`, `:227`; `mapper_utils.py:193` | New submap: uniform, RGB-gradient and edge samples. Existing: alpha holes and depth discrepancy. Backproject `1.005 D`; frustum filter then FAISS neighbor exclusion; Open3D filtering for some datasets. | Zero-gradient distribution and all-missing depth need checks. FAISS L2 returns squared distance but the code directly compares it with `radius`; preserve that effective threshold in reproduction. Rebuilds indexes. |
| Mapping optimization | `mapper.py:139`, `:258` | Sample current/past keyframe, RGB L1 + DSSIM, depth L1, isotropy penalty and `1e-4 NLL`. RGB/depth squared residuals in NLL are detached. Optional depth residual shares RGB variance channels. | Loop runs `iterations+1` evaluations/backward passes and `iterations` optimizer steps. Opacity pruning halfway/end. Map timing excludes insertion and some rendering/logging. |
| Densification/pruning | `gaussian_model.py:141`, `:383`, `:444` | Append parameters and optimizer states; remove low-opacity splats and states. | `densification_postfix` is append bookkeeping; the online mapper does not invoke gradient-based split/clone densification. `grow_submap` freezes SH features after insertion; do not describe online mapping as freely optimizing all colors. |
| Submaps | `gaussian_slam.py:123–238` | Fixed interval or 50°/0.5 m motion trigger; checkpoint previous map; reset active map/keyframe list. Loop corrections rigidly transform saved maps and pose ranges. | Final active submap is not saved after the loop; trajectory includes a tail without exported maps. Gaussian quaternion conventions in rigid correction need an explicit numerical test (model wxyz vs ROMA convention). Retains Gaussian objects for reliability calculations. |
| Descriptor extraction / hloc | `lc.py:139`; `gsr/descriptor.py:16` | hloc `NetVLAD`, VGG16/Pitts30K whitening, 4096D descriptors on RGB keyframes. Self-similarity sets per-query thresholds. | **Current hloc detector is already NetVLAD.** A plain ungated NetVLAD comparator must be distinguished from VarSplat's reliability-gated NetVLAD. No SuperPoint/SuperGlue matching is called here. Descriptor similarity also chooses registration views. |
| Loop detection | `lc.py:122`, `:209` | Per-splat RGB mean variance → median-centered log weight → mean normalization → clamp → opacity-weighted submap scalar r. Multiply cross-descriptor similarities by `clip(rq rdb, .85, 1.20)`; compare with self thresholds and filter temporal submap interval. | No RGB-D-V learned token model in this path. Dense descriptor comparisons scale with query/database keyframes; reliability recomputed. `unsqueeze(0)` broadcasts a leading dimension but final index selection still selects database columns. |
| Registration | `lc.py:526`; `gsr/solver.py:105` | Bidirectional FAISS overlap gate; select top 2 views per side by NetVLAD similarity; independently localize 4 cameras, then inverse-residual-weighted translation and Procrustes rotation averaging. | Deep-copies both Gaussian maps/cameras. Hard-coded overlap threshold 0.2; distance comparison is squared L2 vs 0.1. Two views required per submap. Convergence flags collected but do not determine returned success. |
| Camera localization | `gsr/solver.py:16`; `gsr/loss.py:100`; `pose_utils.py:76` | Up to 100 Adam steps, exponential SE(3) increments, exposure parameters, plateau scheduler; convergence at increment norm <1e-4. Opacity/gradient-masked RGB loss weighted by detached V plus valid depth loss. | Positional-call bug: configured tau is passed as `use_variance`, configured w_limit as tau, and default w_limit remains 3.0. Thus tau=50/w_limit=3 gives effective tau=3, not 50. CPU test confirms Python binding. |
| Graph construction | `lc.py:261` | Identity-initialized correction nodes, identity adjacent edges, registered loop edges. Loads all submaps; diagnostic GT transforms only measure edge errors. | Expensive normals and information construction even for diagnostic/identity calls. Rebuilds graph. No learned verifier. Initial pre-registration overlap can reject real loops with large drift. |
| Pose graph | `lc.py:343` | Open3D Levenberg–Marquardt global optimization, uncertain loop edges, configured edge pruning, anchor node 0. | Information matrix comes from Open3D point-cloud correspondences, **not** registration residual Hessian/covariance. `pgo_max_iterations` is configured but not supplied. Optimization gated by increased loop-edge count. Robust graph treatment is existing prior work. |
| Correction | `gaussian_slam.py:177`, `:206`, `:222` | Apply each submap correction to centers, orientations, represented pose interval and current tracked frame. | Disk reread/rewrite and quaternion-order risk. No Gaussian deformable correction or global BA here. |
| Evaluation | `evaluation/evaluator.py`; `evaluate_trajectory.py` | SE(3) Horn alignment, no scale. TSDF mesh from saved maps; 30,000-step global map refinement; observed-view PSNR, **MS-SSIM** under key `ssim`, Alex LPIPS. ScanNet++ novel views emit PSNR only. | Upstream trajectory routine silently truncates longer GT. `run()` calls reconstruction twice (first unguarded), does not call raw `run_rendering_eval`, and catches some errors. Replica mesh references contain author-local absolute paths. Global refinement supplies HWC tensors to the local CHW SSIM function; must preserve/audit separately. |

### Sigma² and V: actual implementation

`GaussianModel._color_variance` stores three log variances per Gaussian, initialized
to -7; `get_color_variance()` returns their exponentials. This differs from geometric
covariance, which is constructed from scale and rotation. Log variances are serialized
in checkpoints and PLYs, appended with new Gaussians, and pruned with the others.

Mapping/tracking use `src/utils/utils.py:121` → `gaussian_rasterizer`.
Registration uses `src/gsr/renderer.py:24` → `diff_gaussian_rasterization`.
Pinned source submodules:

- mapping: `b7f9982db6d8d7e80c953670181b9d2e721ea8f2`;
- registration: `0a828ff840a2cd60a807b94430277591586d717a`.

In both `cuda_rasterizer/forward.cu` files, with front-to-back weight `w_i=alpha_i T_i`,
the kernel accumulates `M_c=sum(w_i mu_ic)` and `S_c=sum(w_i (sigma²_ic+mu_ic²))` in
double precision for the variance path. It returns `clip(S_c-(M_c+T bg_c)^2,1e-6,0.2)`.
All current callers use black background; the kernel does not add a background second
moment. There is no division by accumulated opacity. V is a 3-channel appearance
variance, not a calibrated depth or pose covariance.

Mapping backward source at `gaussian_rasterizer_var/cuda_rasterizer/backward.cu:563`
reads `dL_dvariance[pix_id]`, doubles its contribution and adds it to all three input
channels. Job 6815 independently confirmed the resulting cross-channel and factor-of-two
derivative errors against finite differences in the installed kernel. The clamp boundaries
still need a separate check. Do not silently fix this in the baseline.

### Config inventory and protocol mismatch

All 27 scene configs plus 4 dataset bases were inspected: 8 Replica, 5 TUM, 9 ScanNet,
5 ScanNet++. The main paper's ScanNet tracking table uses only 0000, 0059, 0106, 0169,
0181 and 0207; 0054/0233/0465 are additional configs, not interchangeable with that
table's six-scene denominator. ScanNet++ scene files cap training at 250 frames.
Its base registration config omits `use_render`, which the registration function reads
with direct indexing. `lc.final` is false in all bases; enabling it exposes additional
unsaved-map/None reliability problems and is an intervention.

The paper describes keyframe ATE. The main evaluator computes all available trajectory
poses. M0 reports all-frame ATE, mapping-frame ATE and exported-keyframe ATE separately,
with explicit IDs. A paper comparison cannot assume these denominators are identical.
The paper's room0 timing table reports optimization-only mapping/tracking per input
frame on A100 80 GB, not full end-to-end runtime. Cluster A100 memory must be reported.
Source: [VarSplat paper, Tables 1–10](https://arxiv.org/html/2603.09673v1).

## Ideas table (M0 infrastructure, not claimed SLAM gains)

| Idea | Hypothesis / expected ATE and time effect | Risk and differentiation | Ablation |
|---|---|---|---|
| Strict coverage-aware metric harness | No intended ATE change; prevent apparent gains caused by missing poses or changing keyframe sets. Modest offline cost. | Standard evaluation hygiene; no novelty claim. Keep paper and full-frame metrics separate. | Upstream truncating evaluator vs strict full-length and explicit exported/mapping-ID reports. |
| Separate wall benchmark from synchronized stage profiler | No intended ATE change; expose actual tracking/mapping/loop/I/O costs. Diagnostic run is slower. | Standard profiling, not a research contribution. Do not sum nested timings or rank variants using profiled runs. | Same seed/config in wall vs stages; quantify profiler overhead and trajectory parity. |
| Trace uncertainty from channels to effective registration settings | Candidate original **experiment design for this fork**: combine channel-wise finite differences, effective call-argument binding and future registration/retained-edge diagnostics. Could identify false attribution to uncertainty. No ATE/runtime prediction until measured. | I did not find this exact diagnostic combination in the six checked papers. That is not evidence of a novel SLAM algorithm; component checks are standard. | Per-channel analytic/finite-difference gradients; unchanged upstream vs one isolated correction at a time, only after baseline capture. |

## Small reviewable code diffs

Additive files only: strict metric helper/tests, independent preflight, opt-in stage
profiler, baseline runner and repetition aggregator. Original `run_slam.py`,
`run_evaluation.py`, `src/` and `configs/` remain byte-equivalent to the pinned source.
The harness bypasses the CLI's seed-zero bug by assigning the requested seed directly
to the resolved configuration; it disables W&B for all measured arms. No algorithm
fix is enabled. `--profile stages` or `benchmark.profile: stages` enables diagnostics;
default `wall` leaves per-stage methods untouched.

## Benchmark table, ablations and next-module gate

See `RESULTS.md`. Baseline reproduction is incomplete. No M1–M8 improvement has been
implemented or promoted. Required repetitions are at least three seeds per scene;
the runner writes `benchmark.json` and the aggregator rejects missing/duplicate seeds,
failed runs and mixed profiling/config/environment conditions.

IMU remains deferred. Future interfaces must carry explicit frame IDs, timestamps,
camera convention, calibration version and optional IMU samples without consulting GT.
Do not add a fake IMU implementation or change RGB-D behavior during M0.

Open decisions before M1: reconcile paper-vs-code keyframe evaluation; choose which
upstream defects are separately corrected after capture;
confirm a train/development/test split accounting for prior exposure to Replica/TUM.
