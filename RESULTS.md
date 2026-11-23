# VarSplat research results

## M0 — source and reproduction status (23 November 2026)

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

**Baseline reproduction is incomplete. No improvement is claimed.**

| Experiment | Data / repetitions | ATE SE(3), cm | SLAM time / memory / rendering | Interpretation |
|---|---|---:|---|---|
| Paper reference | TUM fr1/desk | 1.80 | Not remeasured | [VarSplat paper](https://arxiv.org/html/2603.09673v1), not our result |
| Author saved trajectory, job 6814 | 592 associated poses, one existing checkpoint | 1.8032138 | Not measured | Independent trajectory re-evaluation only; not a fresh SLAM run |
| Same saved trajectory, exported keyframes | 525 poses | 1.8551259 | Not measured | Demonstrates sensitivity to evaluation coverage |
| Fresh upstream baseline | Seeds 0, 1, 2 planned | Pending | Pending | Do not compute deltas from the saved checkpoint |

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
| 6820, 6821 | Submitted | Author-config and repository-config seed-0 baselines, with archive hash and every listed PNG checked before SLAM |

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
