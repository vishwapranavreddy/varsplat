# Literature check — 2026-11-23

25 November additions: [MMG-SLAM](https://github.com/CoolSunflower/MMG-SLAM)
already integrates MambaVision descriptors with Gaussian SLAM; the recovered
candidate explicitly follows its descriptor semantics.
[BayesianGS-SLAM](https://arxiv.org/html/2609.24140v1) combines sensor and
opacity-induced representation uncertainty for RGB/depth prediction, tracking
and keyframe selection. [MonoCal](https://arxiv.org/html/2611.19188v1) separates
scale and shape calibration without ground-truth depth. These materially narrow
potential M1/M5 novelty claims. See the scoped hypotheses and limitations in
[M1_UNCERTAINTY.md](M1_UNCERTAINTY.md).

This is a scoped prior-art screen, not an exhaustive novelty determination. None of
the proposed M1–M8 ideas is claimed original or better on this evidence alone.

| Primary source | Already established | What a differentiated contribution must demonstrate |
|---|---|---|
| [LoopSplat](https://arxiv.org/abs/2408.10154) | RGB-D submaps, Gaussian-space registration, loop closure and robust pose graph optimization. | A new registration/verification/covariance mechanism and measured comparison; merely adding these components is not novel. |
| [SCOUT-SLAM](https://arxiv.org/abs/2609.14634) | Shared network with distinct tracking/mapping uncertainty, multi-view self-supervision and a spatial prior. | Separate our appearance/depth calibration and temporal loop task from shared uncertainty prediction; test on unseen scenes. |
| [UncLe-SLAM](https://arxiv.org/abs/2306.11048) | Self-supervised online pixel-wise depth uncertainty and multi-sensor uncertainty modeling for neural SLAM. | Learned depth-noise fusion is not original by itself. Define calibration target and benefit beyond analytic noise weighting. |
| [CG-SLAM](https://arxiv.org/abs/2403.16095) | Uncertainty-aware Gaussian field, depth uncertainty and primitive selection for efficiency. | Uncertainty-based pruning/selection alone is insufficient. Quantify a distinct criterion and its cost/quality tradeoff. |
| [OverlapMamba](https://arxiv.org/abs/2405.07966) | State-space/Mamba place recognition from LiDAR range views. | Mamba loop recognition is existing work. Temporal RGB-D-V tokens and registration-conditioned supervision would need exact architectural/objective comparisons. |
| [Flash-Mono](https://arxiv.org/abs/2604.03092) | Recurrent feed-forward poses/Gaussians and hidden-state submap descriptors for loop closure. | Hidden states for submap retrieval are not sufficient novelty. Distinguish from a recurrent predictive frontend. |
| [VarSplat](https://arxiv.org/html/2603.09673v1) | Learned per-splat appearance variance, rendered variance and reliability-weighted tracking/registration/loop detection. | Variance-weighted registration (M5c) is already in the starting system. A new residual model or covariance propagation must be specified. |

I did not find the exact combination of temporal RGB-D-V sequence tokens,
registration-success prediction from observed/rendered pairs, and calibrated
registration covariance in these six named comparison papers. This does not establish
novelty. Search broader learned verification, registration confidence, uncertainty-aware
retrieval and robust graph literature before implementing/claiming M5 originality.

Additional search lead: GLC-SLAM, “Robust loop closure for monocular Gaussian splatting
SLAM”, has a publisher abstract describing uncertainty-guided descriptors and rendered
view validation ([publisher](https://www.sciencedirect.com/science/article/abs/pii/S0957417426024942)).
Its complete method needs review before claiming the verifier combination is distinct.

M0 itself is evaluation infrastructure. Its combined diagnostics may be specific to
this fork, but standard gradient tests, coverage audits and profiling are not new SLAM
algorithms. No estimated percentage gain is presented as a result.
