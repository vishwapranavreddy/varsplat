#!/bin/bash
#SBATCH --job-name=varsplat-m0-baseline
#SBATCH --account=mtech
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
cd "$TMPDIR" || exit 1
set -euo pipefail
export M0_SCENE="${1:?full TUM scene name}"
export M0_CONFIG_FLAVOR=repository
export M0_SEED="${2:?seed}"
export M0_PROFILE="${3:-wall}"
case "$M0_SCENE" in
  rgbd_dataset_freiburg1_desk|rgbd_dataset_freiburg1_desk2|rgbd_dataset_freiburg1_room|rgbd_dataset_freiburg2_xyz|rgbd_dataset_freiburg3_long_office_household) ;;
  *) exit 2 ;;
esac
case "$M0_PROFILE" in wall|stages) ;; *) exit 2 ;; esac
[[ "$M0_SEED" =~ ^[0-9]+$ ]] || exit 2
export PROJECT_ROOT="$HOME/projects/varsplat-enhanced"
mkdir -p results code home cache/torch/hub/netvlad cache/torch/hub/checkpoints
cp -r --no-preserve=all "$SLURM_SUBMIT_DIR/src" "$SLURM_SUBMIT_DIR/configs" "$SLURM_SUBMIT_DIR/tools" code/
cp --no-preserve=all "$SLURM_SUBMIT_DIR/run_slam.py" "$SLURM_SUBMIT_DIR/run_evaluation.py" "$SLURM_SUBMIT_DIR/requirements.txt" "$SLURM_SUBMIT_DIR/.gitmodules" code/
cp -r --no-preserve=all "$HOME/job_results/6817/runtime" runtime
cp -r --no-preserve=all "$HOME/projects/varsplat-m0-evalpins-20261124/wheels" eval_wheels
ARCHIVE="$HOME/projects/varsplat-m0-tum-inputs-20261124/$M0_SCENE.tgz"
if [ "$M0_SCENE" = rgbd_dataset_freiburg1_desk ]; then
  ARCHIVE="$HOME/projects/varsplat-m0-inputs-20261123/$M0_SCENE.tgz"
fi
python3 code/tools/analysis/stage_scene.py --dataset tum_rgbd --scene "$M0_SCENE" --archive "$ARCHIVE" --target "$TMPDIR/input" --output results/input_integrity.json
cp --no-preserve=all "$PROJECT_ROOT/cache/torch/hub/netvlad/VGG16-NetVLAD-Pitts30K.mat" cache/torch/hub/netvlad/
cp --no-preserve=all "$HOME/projects/varsplat-m0-environment-20261123/models/alexnet-owt-7be5be79.pth" cache/torch/hub/checkpoints/
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TMPDIR/mpl"
export MPLBACKEND=Agg
export PIP_CACHE_DIR="$TMPDIR/pip-cache"
export XDG_CACHE_HOME="$TMPDIR/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Monitor driver-reported process memory, including non-PyTorch allocations.
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,used_gpu_memory --format=csv -lms 500 > results/gpu_process_memory.csv 2> results/gpu_monitor.err &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
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
CONFIG="configs/TUM_RGBD/$M0_SCENE.yaml"
python -B tools/m0/check_environment.py --output "$TMPDIR/results/environment.json"
python -B tools/m0/run.py "$CONFIG" --input "$TMPDIR/input" --output "$TMPDIR/results/run" --seed "$M0_SEED" --profile "$M0_PROFILE" --evaluation upstream
if [ "$M0_PROFILE" = wall ]; then
python -B tools/analysis/fixed_frames.py --run-dir "$TMPDIR/results/run" --input "$TMPDIR/input" --output "$TMPDIR/results/fixed_frames.json"
fi
' > results/console.log 2>&1
