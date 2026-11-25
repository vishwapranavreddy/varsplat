# Existing Mamba system: recovered implementation and evidence

Recovery date: 25 November 2026. This is an audit of prior work, not a newly
measured improvement. The current upstream-preserving fork remains the development
repository. No older estimator files were copied over its `src/`.

## Analysis

The other chat's latest frozen candidate is
`~/projects/vslam-modular/full_variance_mamba_faiss_v1/source`, with recorded source
manifest identity `8b543d8d95a1bc1de3aefd7de3b0336edc623cb4153ba1b8530fb3c6be5e3cf0`.
Its 120 Python/configuration files were hashed on the cluster. The older local
working copy differs in ten paths, including the live loop adapter and FAISS
index. Therefore the recovered frozen source, rather than that working copy,
is the reference for this audit. A separate byte-exact snapshot and retrieval
manifest are in `../../recovered_mamba_20261125/`.

| Stage | Recovered execution path | Behavior and limitation |
|---|---|---|
| Input | `src/entities/datasets.py`, `loop_ext/rgbd.py` | Candidate uses registered/calibrated RGB-D decoding. This is a whole-system difference from public VarSplat, not a Mamba-only ablation. |
| Tracker/map selection | `entities/gaussian_slam.py:47,69` | Original tracker and mapper remain selectable. Guarded tracker/reliable mapper classes existing in the source do not prove they were enabled; inspect each resolved experiment configuration. |
| Loop routing | `entities/gaussian_slam.py:75–92` | `lc.backend: reserved_sensor` selects a sensor registration backend; `varsplat` retains Gaussian-space registration. These are different experimental arms. |
| Descriptor | `loop_ext/live_loop.py:39`, `keyframe_descriptor.py:52`, `mmg_descriptor.py:35` | Frozen MambaVision-T, 224x224 RGB preprocessing, normalized 640D pooled features, content-addressed cache. No temporal hidden state or RGB-D-V token fusion. MMG-SLAM descriptor semantics are explicitly acknowledged in the source. |
| Retrieval | `loop_ext/faiss_submap_index.py:6` | Exact CPU `IndexFlatIP`, one FAISS thread, increasing shortlist until eligible unique-submap top-K is provably complete. Rebuilds the database on each submap update; no learned index. |
| View selection | `loop_ext/view_reservation.py`, `live_loop.py:85` | At most nine sampled keyframes per submap with disjoint fitting, admission and graph-observer roles. Retrieval views are not themselves proof of successful loop correction. |
| Registration | `loop_ext/sensor_registration.py:107,119`, `geometry.py:54` | RGB-D feature geometry, correction consensus, balanced Huber refinement, reserved bidirectional depth checks. The active sensor route does not perform Gaussian image registration or consume rendered appearance variance. |
| Uncertainty | `entities/mapper.py`, corrected native mapping kernel; `loop_ext/geometry.py:62` | Per-splat appearance variance remains in mapping/tracking. Sensor registration uses an analytic depth-noise model. Appearance variance and metric depth noise are not interchangeable covariances. |
| Pose graph | `loop_ext/reserved_backend.py`, `graph_optimization.py`, `graph_guard.py` | Robust proposals, frozen observer identities and acceptance checks. `loop_information_scale=0.25` and topology preselection in the frozen plan are heuristic policies, not measured covariance calibration. |
| Odometry | `loop_ext/odometry_backbone.py:18` | Records relative motions before a new loop is applied; avoids reinterpreting corrected poses as independent odometry measurements. It is a useful future IMU integration boundary, not an IMU implementation. |

The candidate's old GPU finite-difference proof reports channel-correct appearance
variance derivatives and zero gradient above the upper clamp. The public baseline
proof reports cross-channel/doubled gradients and a nonzero saturated derivative.
These fixes were bundled with other changes, so their isolated ATE effect is unknown.

## Measured prior results, kept separate from M0

| Existing campaign | Public VarSplat | Mamba candidate | Interpretation |
|---|---:|---:|---|
| Eight Replica scenes, seed 1, mean ATE | 0.251415 cm | 0.317656 cm | Candidate mean is 26.35% worse; wins on 3/8 scenes |
| Same eight scenes, mean core wall time | 8957.500 s | 5255.047 s | Candidate uses 41.33% less time; includes other changes |
| Same scenes, mean loop wall time | 3937.629 s | 186.587 s | Large reduction; only two scenes apply accepted candidate correction events |
| TUM fr1/desk, seed 1, jobs 6791/6793, ATE | 2.396169 cm | 1.752019 cm | Retrieved completed reports, 592 frames; single seed |
| Same desk pair, core wall time | 2624.910 s | 2128.114 s | Same older protocol; not the new M0 end-to-end/evaluation protocol |
| Same desk pair, loop wall time | 571.642 s | 23.536 s | 7 versus 4 correction events; recall not established by event count |

Replica evidence: the other chat's `outputs/job_review_v21/analysis.md`.
Desk evidence: recovered `jobs/6791/` and `jobs/6793/` summaries, trials and native
proofs. Desk summaries agree with their saved independent trajectory-metric fields;
this recovery did not re-evaluate trajectories or render maps. No quality numbers
or multi-seed confidence are inferred from these summaries.

## Module order and ablations

Continue M0 baseline reproduction and profiling first. M1 isolates the uncertainty
gradient correction, depth-noise model and learned calibration. M2–M4 isolate
tracking, Gaussian lifecycle and submap scheduling. M5 then compares the recovered
RGB-only Mamba descriptor, original reliability-gated NetVLAD, an explicitly
ungated NetVLAD control, DINOv2 and the proposed temporal RGB-D-V model under the
same verifier and candidate budget. Do not attribute the recovered bundle's speed
to Mamba alone or relax graph admission before measuring where candidates fail.

For each candidate, retain upstream defaults, record one flag per change, run at
least three paired seeds, and promote only measured ATE/runtime wins. Previous
Replica/TUM scores are development evidence; they cannot be used to select a model
and simultaneously claim untouched final-test generalization.

## Open question

The train/development/final-test scene split is pending user clarification. The
recommended split reserves untouched ScanNet/ScanNet++ scenes for final testing;
their dataset access is still unresolved. M0 and source diagnostics can continue.
