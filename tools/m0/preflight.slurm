#!/bin/bash
#SBATCH --job-name=varsplat-m0-check
#SBATCH --account=mtech
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
cd "$TMPDIR" || exit 1
set -euo pipefail
mkdir -p results code
MODE="${1:?runtime or variance}"
SOURCE="$SLURM_SUBMIT_DIR"
ROOT="$HOME/projects/varsplat-enhanced"
cp -r --no-preserve=all "$SOURCE/src" "$SOURCE/configs" "$SOURCE/tools" code/
export PROJECT_ROOT="$ROOT"
export M0_MODE="$MODE"
export M0_RESULT="$TMPDIR/results/$MODE.json"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TMPDIR/mpl"
export XDG_CACHE_HOME="$TMPDIR/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
apptainer exec --nv --bind "$TMPDIR:$TMPDIR" /userhome/containers/jupyter.sif bash -c '
set -euo pipefail
source "$PROJECT_ROOT/scripts/env.sh"
cd "$TMPDIR/code"
export PYTHONPATH="$PWD:$PROJECT_ROOT/thirdparty/Hierarchical-Localization"
python -B tools/m0/preflight.py "$M0_MODE" --output "$M0_RESULT"
' > results/console.log 2>&1
