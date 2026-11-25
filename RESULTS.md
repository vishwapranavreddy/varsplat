# VarSplat research results

## 25 November continuation: Mamba recovery and M0 repetitions

Recovered the frozen MambaVision-T + FAISS candidate and completed desk receipts
from the other chat. See [MAMBA_RECOVERY.md](docs/MAMBA_RECOVERY.md) for source
identity, active execution paths and prior positive/negative evidence. Those older
bundled experiments are not merged into the current baseline repetitions.

| Current full fr1/desk baseline | Seed | Job | SE3 ATE, cm | SLAM wall, min |
|---|---:|---|---:|---:|
| Author configuration | 0 | 6826 | 2.493786 | 56.2076 |
| Author configuration | 1 | 6830 | 2.395647 | 53.9849 |
| Repository configuration | 0 | 6827 | 1.493297 | 45.6697 |
| Repository configuration | 1 | 6831 | 1.662930 | 45.3110 |

Both seed-1 jobs completed with exit code zero and 592 poses. Their native
rendering coverage is 540/520 frames, so native quality means are not pooled.
Seed-2 jobs **6833/6834** were submitted after the account queue was checked empty.
Their additive launcher runs unchanged SLAM first, then fixed-frame replays of
both seed 2 and the corresponding seed-1 saved map. Replay time is outside SLAM
timing; full batch time includes these extra audits.

New `tools/analysis/aggregate_replays.py` joins reports by exact baseline hash,
checks every frame and native-subset parity, and rejects mixed evaluator/weight
identities. `uncertainty_audit.py` tests appearance calibration and scalar-weight
invariance without changing the estimator. All **28 small CPU tests passed**.
Both new Slurm launchers passed `bash -n` on the cluster before deployment.

[M1_UNCERTAINTY.md](docs/M1_UNCERTAINTY.md) records analysis, four candidate
interventions, prior-art checks, ablations and unresolved scene-split requirements.
It is preparation only: M0 is incomplete, no M1 estimator flag is enabled, and no
new accuracy/runtime gain or novelty is claimed. Original src/config/native paths
remain intact. A three-seed result and multi-scene reproduction are still required.

## M0 — source and reproduction status (24 November 2026)

Upstream: `4a21a0aaade40b7c42d47181108e49a8a0e695f2`. Local research
branch: `research/m0-reproduction`. Hosted fork:
[vishwapranavreddy/varsplat](https://github.com/vishwapranavreddy/varsplat), verified
as a fork of `anhthuan1999/varsplat` through the authenticated GitHub API.
Source analysis and ideas are in [M0_CODE_MAP.md](docs/M0_CODE_MAP.md).
Original source, configs and native kernels remain unchanged.
The older clean cluster checkout at commit `65e3623b9e0e602be9176e41bc914bde8bf48411`
and current upstream have identical Git tree `5cd3104976094fa33c2601e17becd314efa83529`.
Commit IDs alone therefore overstate their source difference. The separate older
`varsplat-enhanced` working tree is modified; its runs are not pristine baseline evidence.

**Two full seed-0 baselines are complete. Multi-seed/multi-scene reproduction is
incomplete. No algorithm improvement is claimed.**

| Experiment | Data / repetitions | ATE SE(3), cm | SLAM time / memory / rendering | Interpretation |
|---|---|---:|---|---|
| Paper reference | TUM fr1/desk | 1.80 | Not remeasured | [VarSplat paper](https://arxiv.org/html/2603.09673v1), not our result |
| Author saved trajectory, job 6814 | 592 associated poses, one existing checkpoint | 1.8032138 | Not measured | Independent trajectory re-evaluation only; not a fresh SLAM run |
| Same saved trajectory, exported keyframes | 525 poses | 1.8551259 | Not measured | Demonstrates sensitivity to evaluation coverage |
| Fresh author-config baseline, job 6826 | fr1/desk, seed 0, 592 poses | 2.493786 | See measured tables below | Completed; one seed is insufficient to establish paper reproduction |
| Fresh repository-config baseline, job 6827 | fr1/desk, seed 0, 592 poses | 1.493297 | See measured tables below | Completed; unchanged upstream algorithm, not a proposed improvement |

### Completed seed-0 measurements

All measurements below use the unchanged source and original rasterizers on A100
PCIe 40 GB. Seeds 1 and 2 are required before mean/std or acceptance decisions.
The strict SE(3) ATE agrees with the independent upstream evaluator within 1e-9 m
for both runs, with exactly 592 compared poses.

| Configuration / job | SLAM wall, s | s/input frame | Offline evaluation, s | Full batch elapsed, s | Sampled process GPU maximum, GiB | PyTorch allocated / reserved maximum, GiB |
|---|---:|---:|---:|---:|---:|---:|
| Author / 6826 | 3372.453 | 5.696711 | 425.621 | 3870 | 3.554688 | 1.192357 / 1.511719 |
| Repository / 6827 | 2740.182 | 4.628685 | 408.462 | 3216 | 3.533203 | 1.200126 / 1.500000 |

SLAM wall includes tracking, mapping, loop closure, pose correction and upstream
I/O/diagnostics. Batch elapsed includes staging and preflight as well. PyTorch's
reported SLAM maximum includes initialization because its peak counter starts before
initialization; driver samples are restricted to the exact SLAM interval and PID.
Driver sampling had maximum gaps of 0.509 / 0.511 seconds. It is a sampled maximum,
not an exact peak and not the sum of memory used by both jobs.

| Native upstream rendering protocol | Frames | PSNR, dB | MS-SSIM | Alex LPIPS | Depth L1, m |
|---|---:|---:|---:|---:|---:|
| Author config, seed 0 | 520 | 21.931347 | 0.846407 | 0.311453 | 0.191948 |
| Repository config, seed 0 | 523 | 22.538380 | 0.864621 | 0.287795 | 0.190356 |

These native scores cover different exported frame IDs. They are retained for
paper-protocol auditing, not pooled as a matched-coverage comparison.

| Supplemental all-associated-frame protocol | Replay job | Frames | PSNR, dB | MS-SSIM | Alex LPIPS | Depth L1, m |
|---|---|---:|---:|---:|---:|---:|
| Author config, seed 0 | 6828 | 592 | 22.116970 | 0.849789 | 0.314283 | 0.173973 |
| Repository config, seed 0 | 6829 | 592 | 22.595683 | 0.864307 | 0.294486 | 0.173177 |

The supplemental replay reloads the saved global PLY without retraining. Its native
frame subset reproduced each upstream mean: maximum absolute discrepancy across
both runs was 5.69e-7 dB PSNR, 7.34e-7 MS-SSIM, 1.09e-8 LPIPS and zero depth L1.
Full replay jobs took 64 / 57 seconds; timed rendering/metrics took 33.022 / 27.909 s.
These times are separate from the baseline SLAM/evaluation times. This is observed-view
quality, not a held-out-scene or novel-view generalization result.

Evidence: byte-exact `../cluster_preflight/6826/` and `6827/` benchmark, upstream
metric and memory files; `6828/fixed_frames.json` and `6829/fixed_frames.json` contain
per-frame quality, baseline/model/input hashes and parity checks. Report hashes in
the replay were checked against the downloaded baseline bytes. Earlier text-transfer
newline conversion was corrected in the retrieval helper; numerical results did not
change. Seed-1 jobs **6830/6831** were submitted with the same frozen Python source,
native environment and input bytes, after the account queue was verified empty.

Raw evidence: `../cluster_preflight/6814/runtime.json` and
`../cluster_preflight/6815/variance.json`. Author trajectory SHA-256:
`ecf1c1eaecb489dd5af7a762292d14686b34d933d6c4f48a60387926e0605e65`.
These measurements used A100 PCIe **40 GB**, PyTorch 2.6.0+cu126, Python 3.10.21,
NumPy 1.26.4. The paper's stated 80 GB A100 is a hardware mismatch;
runtime comparisons must use our own paired reproduction.

## Diagnostics, including unsuccessful attempts

| Jobs | Outcome | Meaning / next action |
|---|---|---|
| 6812, 6813 | Failed before compute, scratch copy metadata permission errors | Replaced `cp -a` with copy without preserved metadata; retained failure receipts |
| 6814 | Diagnostic completed in 85 s; environment BLOCKED | Missing `pycg`, missing reconstruction package on path, CPU-only FAISS; checkpoint ATE measured |
| 6815 | Diagnostic completed in 45 s; gradient MISMATCH | Original installed mapping rasterizer fails all three channel finite-difference checks |
| 6816 | Setup failed in 15 s | Compute-node PyPI TLS hostname mismatch; no TLS bypass. Stage hash-verified wheels through working login HTTPS, then install offline in scratch |
| 6817 | Environment check passed in 47 s | Isolated official FAISS GPU 1.16.0 + PyCG; NumPy and Torch unchanged. GPU NumPy/Torch searches match exact L2 reference. Imports pass; this did not validate dataset images |
| 6818, 6819 | Failed before tracking, 38 s each | Old dataset retained lists/poses but most images had been retired. Recovered verified existing local dataset backup; no new dataset download |
| 6820, 6821 | Failed at first PGO diagnostics (18:45 / 15:27 job elapsed) | Both reached tracking, mapping and loop registration. Installed evo 1.31.1 lacks `trajectory.align_trajectory`, which upstream calls. These are incomplete runs, not baselines |

24 November follow-up: stage the repository-pinned `evo==1.11.0` and
`matplotlib==3.5.1` in a separate overlay, keeping old environments intact.
Preflight now executes the actual upstream trajectory alignment and plot export,
two original refinement iterations on a small synthetic fixture, and offline
LPIPS/MS-SSIM. A full run starts only if those checks pass. No algorithm change.
Jobs 6822/6823 stopped in preflight because evo's own default selected `TkAgg`,
overriding `MPLBACKEND=Agg`. Configure `plot_backend=Agg` in evo's isolated scratch
settings before imports. Jobs 6824/6825 then stopped in preflight because the checker
assumed an `evo.__version__` attribute absent in 1.11.0. The checker now reads installed
distribution metadata and records the imported module path. Replacement seed-0 jobs
6826 (author config) and 6827 (repository config) passed all expanded preflight checks
and reached tracking (frames 159 / 194 at the 14-minute check); no SLAM source changes.
The checks cover actual upstream Evo plotting, two synthetic refinement steps,
offline AlexNet LPIPS and MS-SSIM, and both NumPy/Torch GPU FAISS operations.
Synthetic smoke-test values are not benchmark results.
Both jobs subsequently completed full SLAM and upstream evaluation; measured
seed-0 results and the successful fixed-frame audits are recorded above.
Whole-home usage was remeasured at 48,782,098,432 bytes before submission; 12 GB of
return space is reserved for the pair, below the 200 GB hard cap.

The author checkpoint config is **not identical** to current repository defaults:

| Parameter | Author fr1/desk checkpoint | Pinned repository fr1/desk |
|---|---:|---:|
| `var.tau_track` | 20 | 50 |
| `var.tau_registration` | 20 | 50 |
| `var.tau_lc` | 5 | 10 |

Both configurations are evaluated separately. The source's positional registration
argument defect means the configured registration threshold does not bind as intended.
Do not pool these arms or silently replace one configuration with the other.
The recovered 344,011,403-byte TUM archive matches the previous cleanup receipt,
SHA-256 `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`.
The old incomplete input directory is untouched; extraction occurs only in job scratch.

Kernel test: one splat, black color, 32×32 render, center pixel, log-variances
[-4, -3.5, -3], centered finite difference step 0.01. Rows differentiate each
output variance channel with respect to all three log-variances:

| Output | Autograd gradient (R,G,B) | Finite difference (R,G,B) |
|---|---|---|
| R | (0.01792184, 0.02954812, 0.04871661) | (0.00896119, 0, 0) |
| G | (0, 0, 0) | (0, 0.01477427, 0) |
| B | (0, 0, 0) | (0, 0, 0.02435865) |

This establishes a derivative mismatch in the installed mapping extension, not an
ATE improvement. Both cluster rasterizer source checkouts match the pinned upstream
submodule revisions and are clean. Native binary hashes are in the runtime report.
Preserve the original kernel for baseline; evaluate corrections separately in M1.
The pinned `cuda_rasterizer/backward.cu` explicitly uses the first-channel adjoint
for every channel and multiplies its contribution by 2, explaining both the cross-channel
leakage and the doubled red derivative in this test.

Local validation: 14 unittest checks passed (strict SE(3), seed aggregation, source/config/input
comparability, non-finite values, profiler failure propagation, exact-process memory
attribution and time-window coverage). Python compilation and Bash syntax checks passed.
Two additional CPU checks validate fixed-frame metric coverage and rejection of
missing/duplicate/non-finite measurements (16 total checks). The supplemental GPU
replay passed native-subset parity on both completed seed-0 baselines, as detailed
above. No quality gain is claimed from this evaluation-tool addition.
Three staging-boundary checks also pass (19 CPU checks total). Input restoration,
Replica archive contents and unresolved reference assets are tracked in
[M0_INPUTS.md](docs/M0_INPUTS.md). TUM fr1/desk2 is now checksum-verified; the three
other transfers resumed after the SSH output pipe disconnected during the approval
interruption. They log persistently on the cluster and remain unverified until their
final checksum passes.
Runtime packages are recorded with import-path precedence in new baseline reports;
the older environment diagnostic's flat distribution listing can show a shadowed version
(e.g. OmegaConf) and must not be treated as the imported module version.

## Ablations and acceptance gate

- Metric coverage: full vs exported-keyframe author ATE measured above; not an algorithm ablation.
- Wall vs synchronized profiling: pending paired run; synchronized stages are inclusive and must not be summed.
- Corrected variance gradients / registration arguments: not implemented; blocked on fresh baseline capture.
- Every baseline/variant needs ≥3 seeds, full pose coverage, identical input bytes,
  source/environment provenance, quality metrics, and total wall time.
- No learned component has been trained or accepted in this campaign.

## Cluster policy applied

The newer CSE JupyterHub guide and live `mtech` QoS permit two running/submitted
jobs per account, including interactive sessions. Older manual text said one running;
the current live limit is two. Use at most two, one GPU per independent run.
First executable Slurm line is `cd "$TMPDIR" || exit 1`; compute and outputs remain
in scratch. `$TMPDIR/results` is automatically returned to `~/job_results/JOBID`.
No duplicate manual copyback. Home storage measured 47,413,547,008 bytes before this
campaign. Rules: 50 GB soft, 15-day grace, 200 GB hard; reserve return space before
submitting. GPU-hours are unrestricted by the user, not by scheduler policy.
No existing experiments, environments or datasets are deleted or modified.

## Open questions before M1

Paper keyframe convention; approved unseen-scene split after prior
Replica/TUM experiments. Dataset availability for ScanNet/ScanNet++ remains to be
audited. These do not prevent the current TUM baseline preparation.
