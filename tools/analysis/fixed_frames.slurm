#!/bin/bash
#SBATCH --job-name=varsplat-m0-coverage
#SBATCH --account=mtech
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
cd "$TMPDIR" || exit 1
set -euo pipefail
export M0_BASELINE_JOB="${1:?completed baseline job ID}"
[[ "$M0_BASELINE_JOB" =~ ^[0-9]+$ ]] || exit 2
export PROJECT_ROOT="$HOME/projects/varsplat-enhanced"
export BASELINE_RUN="$HOME/job_results/$M0_BASELINE_JOB/run"
mkdir -p results code home cache/torch/hub/checkpoints replay
cp -r --no-preserve=all "$SLURM_SUBMIT_DIR/src" "$SLURM_SUBMIT_DIR/configs" "$SLURM_SUBMIT_DIR/tools" code/
cp --no-preserve=all "$SLURM_SUBMIT_DIR/run_slam.py" "$SLURM_SUBMIT_DIR/run_evaluation.py" "$SLURM_SUBMIT_DIR/requirements.txt" "$SLURM_SUBMIT_DIR/.gitmodules" code/
cp -r --no-preserve=all "$HOME/job_results/6817/runtime" runtime
cp -r --no-preserve=all "$HOME/projects/varsplat-m0-evalpins-20261124/wheels" eval_wheels
# This launcher is deliberately limited to the verified fr1/desk input archive.
python3 - <<'PY'
import json, os, shutil
from pathlib import Path
source = Path(os.environ['BASELINE_RUN'])
report = json.loads((source / 'benchmark.json').read_text())
if report['status'] != 'completed' or report['dataset'] != 'tum_rgbd' or report['scene'] != 'rgbd_dataset_freiburg1_desk':
    raise ValueError('Require a completed fr1/desk baseline')
for name in ['benchmark.json', 'estimated_c2w.ckpt', report['scene'] + '_global_splats.ply']:
    shutil.copyfile(source / name, Path('replay') / name)
PY
python3 code/tools/m0/stage_tum.py "$HOME/projects/varsplat-m0-inputs-20261123/rgbd_dataset_freiburg1_desk.tgz" "$TMPDIR/input" --output results/input_integrity.json
cp --no-preserve=all "$HOME/projects/varsplat-m0-environment-20261123/models/alexnet-owt-7be5be79.pth" cache/torch/hub/checkpoints/
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TMPDIR/mpl"
export MPLBACKEND=Agg
export PIP_CACHE_DIR="$TMPDIR/pip-cache"
export XDG_CACHE_HOME="$TMPDIR/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
apptainer exec --nv --home "$TMPDIR/home" --bind "$TMPDIR:$TMPDIR" --bind "$PROJECT_ROOT:$PROJECT_ROOT:ro" /userhome/containers/jupyter.sif bash -c '
set -euo pipefail
source "$PROJECT_ROOT/scripts/env.sh"
cd "$TMPDIR/code"
python -m pip install --no-index --find-links "$TMPDIR/eval_wheels" --no-build-isolation --no-deps \
  --target "$TMPDIR/eval_pins" --report "$TMPDIR/results/eval_pins_install.json" evo==1.11.0 matplotlib==3.5.1
export TORCH_HOME="$TMPDIR/cache/torch"
export HF_HOME="$TMPDIR/cache/huggingface"
export PYTHONPATH="$PWD:$TMPDIR/eval_pins:$TMPDIR/runtime:$PROJECT_ROOT/thirdparty/Hierarchical-Localization:$PROJECT_ROOT/thirdparty/evaluate_3d_reconstruction_lib"
python -B tools/m0/configure_plotting.py
python -B tools/analysis/fixed_frames.py --run-dir "$TMPDIR/replay" --input "$TMPDIR/input" --output "$TMPDIR/results/fixed_frames.json"
' > results/console.log 2>&1
