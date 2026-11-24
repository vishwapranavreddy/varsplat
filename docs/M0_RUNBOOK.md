# M0 execution and interpretation

The assistant runs the cluster campaign. These commands make each run reproducible;
they are not a request for the user to run the work manually.

## Cluster rules and staging

Use account `mtech`, partition `gpu-A100`, one A100 and four CPUs per run. Current
QoS permits two submitted/running account jobs, including Jupyter. Check the whole
account queue before every submission. Do not prequeue a third run or cancel another
chat's job to obtain a slot. GPU-hour budget is unrestricted by the user.

Home policy is 50 GB soft with 15-day grace, 200 GB hard. Check usage and outstanding
return reservations, including other work, before expanding the campaign. The first
two full attempts reserve 12 GB combined for returned artifacts; that is a conservative
reservation, not measured output size. No retention deletion is part of this campaign.

Every batch script starts execution in `$TMPDIR`. Inputs are extracted there, model
weights and the isolated dependency overlay are staged there, and container home/caches
are redirected there through Apptainer. Original project dependencies/source are mounted
read-only. Only `results/` returns automatically to `~/job_results/JOBID`.

## Current pinned assets

- Source: upstream `4a21a0aaade40b7c42d47181108e49a8a0e695f2`.
- Python 3.10.21 / Torch 2.6.0+cu126 / NumPy 1.26.4 / A100 PCIe 40 GB.
- FAISS GPU 1.16.0 overlay: `~/job_results/6817/runtime`; GPU search validated.
- Evaluation pins: `evo==1.11.0`, `matplotlib==3.5.1`, installed offline into each
  job's separate scratch overlay. The older evo 1.31.1 failed at the first PGO diagnostic.
- Baseline native extensions remain unchanged, with binary hashes in each report.
- NetVLAD official checkpoint from the existing cache; SHA-256 recorded.
- AlexNet metric weights staged through verified HTTPS and official torchvision hash
  prefix; full SHA-256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
- TUM fr1/desk archive restored from the other chat's existing verified local backup:
  `~/projects/varsplat-m0-inputs-20261123/rgbd_dataset_freiburg1_desk.tgz`.

No local package installation or dataset download was performed. Dependency downloads
occurred on the cluster login node; installation occurred in an allocated job. Compute-node
internet failed TLS hostname verification, so later jobs install/use packages offline.

## Exact commands

Local CPU checks, from this repository:

```powershell
python -m unittest tools.m0.test_metrics tools.m0.test_aggregate tools.analysis.test_memory -v
python -m compileall -q tools/m0
```

The two source-preserving configurations have different uncertainty thresholds. The
seed-0 deployment directory is `~/projects/varsplat-m0-baselines-20261124-r2`.
Seeds 0, 1 and 2 must be run separately for each configuration. For example, **after
both account slots become free and storage is checked**:

```bash
cd ~/projects/varsplat-m0-baselines-20261124-r2
squeue -u "$USER"
sbatch --output='%j.out' --error='%j.err' tools/m0/baseline.slurm author 1 wall
sbatch --output='%j.out' --error='%j.err' tools/m0/baseline.slurm repository 1 wall
```

Do not repeat a submission when its response is uncertain. Reconcile `squeue`, `sacct`,
the submission receipt and the Slurm comment first. Seed-0 jobs 6826/6827 completed;
their fixed-frame replay jobs 6828/6829 passed parity. Seed-1 jobs 6830/6831 are running
from `~/projects/varsplat-m0-seed1-20261124`, with byte-identical baseline Python
source. Do not repeat those submissions with the example commands.

The batch script executes the following inside the staged repository/environment:

```bash
python -B tools/m0/run.py "$CONFIG" \
  --input "$TMPDIR/input" --output "$TMPDIR/results/run" \
  --seed "$M0_SEED" --profile "$M0_PROFILE" --evaluation upstream
```

`$CONFIG` is either the untouched repository scene YAML or a scratch copy of the
author checkpoint's saved YAML. Only input/output paths, explicit requested seed,
and W&B disabling are overridden. `--profile stages` is a separate diagnostic run,
never pooled with wall benchmarks. `--evaluation raw` evaluates unrefined saved submaps;
it is **not** a replacement for the paper's refined global rendering results.

Example aggregation after three successful, comparable reports are available:

```bash
python tools/m0/aggregate.py \
  /path/to/seed0/benchmark.json /path/to/seed1/benchmark.json /path/to/seed2/benchmark.json \
  --expected-seeds 0 1 2 --output /path/to/new-summary.json
```

Use the actual returned job paths. Never use reports from different configurations,
source versions, GPU models or profiling modes in one aggregation.

## Output and metric semantics

Each `~/job_results/JOBID/` contains `environment.json`, `input_integrity.json`,
`console.log`, driver process-memory samples, and `run/benchmark.json`. The latter
records status/failures, seed, resolved config, source hashes, native hashes, package
versions, input-content hash, associations, GPU details, timings and quality metrics.

Required successful report shape:

```json
{
  "status": "completed",
  "slam_complete": true,
  "profile": "wall",
  "seed": 0,
  "slam_wall_seconds": 0.0,
  "slam_seconds_per_input_frame": 0.0,
  "slam_peak_allocated_bytes": 0,
  "slam_peak_reserved_bytes": 0,
  "metrics": {
    "ate_all_frames": {"rmse_m": 0.0, "alignment": "SE3", "scale": 1.0},
    "ate_mapping_keyframes": {"rmse_m": 0.0},
    "rendering": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "num_renders": 0}
  }
}
```

Zeros above are schema placeholders, **not results**. Aggregates report mean and
sample standard deviation (`ddof=1`). No scale fitting, dropped poses or silent
trajectory truncation is allowed. Input hashing is outside timed SLAM. SLAM wall
includes tracking, mapping, loops, correction and upstream I/O/visualizations;
initialization and offline evaluation are reported separately. `runner_wall_seconds`
includes imports, initialization, hashing and evaluation after initial manifest setup;
it is not complete job elapsed time. Use Slurm elapsed time for staging/job overhead.

PyTorch peaks exclude FAISS/native allocations. Driver CSV samples at 500 ms must
be filtered to the report's PID and the recorded SLAM time interval, and labelled a
**sampled** process-memory maximum rather than an exact peak. Shared-device memory
must never be attributed to one run. Stage timings are inclusive/nested, not additive.

Upstream calls its rendering metric `ssim`, but the implementation is **MS-SSIM**.
Rendering covers exported submap keyframes and omits the final unsaved active submap.
Exact rendered IDs and counts are recorded. Aggregation rejects different rendered
coverage; establish a common evaluation frame set before comparing such runs. Evaluation
exceptions are logged as failures, even when SLAM finished. Never relabel a failed
quality evaluation as a fully reproduced baseline.

### Supplemental fixed-frame audit

`tools/analysis/fixed_frames.py` evaluates every associated observed input frame from
the saved, upstream-refined global PLY. This is a separate coverage protocol, not the
paper's native exported-keyframe score and not an unseen-view generalization test.
It does not retrain the map or change the saved trajectory. It verifies input bytes,
association count, pose ATE, source/native/package identity and offline metric weights.
The subset corresponding to upstream exported frames must reproduce the original
quality means within declared tolerances (PSNR 0.001 dB; other metrics 0.00001).
Per-frame measurements make coverage differences reviewable. Parity failure is a
failure, not permission to silently relax the tolerance.

After a baseline has **completed** and an account slot is free, its GPU replay command is:

```bash
python -B tools/analysis/fixed_frames.py \
  --run-dir /scratch/path/to/copied/completed/run \
  --input /scratch/path/to/verified/associated/dataset \
  --output /scratch/path/to/results/fixed_frames.json
```

The accompanying `fixed_frames.slurm` stages the verified fr1/desk assets and accepts
one completed baseline job ID. Deploy it with the same frozen source as the baseline.
It passed CPU coverage checks, Bash syntax validation and full GPU replay/parity
on seed-0 jobs 6826/6827. Do not execute it concurrently on a GPU running a timed baseline. Keep its
wall time separate from SLAM and from the original upstream evaluation.

## Remaining M0 work

Finish the two configuration arms and seed repetitions; measure profiling overhead;
independently verify quality outputs and memory samples; expand to all paper scenes
after input/space checks; resolve upstream evaluation defects as separately labelled
protocol corrections if needed. Only then begin measured M1 ablations.
