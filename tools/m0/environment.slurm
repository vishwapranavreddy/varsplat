#!/bin/bash
#SBATCH --job-name=varsplat-m0-env
#SBATCH --account=mtech
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
cd "$TMPDIR" || exit 1
set -euo pipefail
mkdir -p results code home cache/torch/hub/checkpoints cache/torch/hub/netvlad
cp -r --no-preserve=all "$SLURM_SUBMIT_DIR/src" "$SLURM_SUBMIT_DIR/configs" "$SLURM_SUBMIT_DIR/tools" code/
cp -r --no-preserve=all "$HOME/projects/varsplat-m0-environment-20261123/wheels" wheels
cp -r --no-preserve=all "$HOME/projects/varsplat-m0-evalpins-20261124/wheels" eval_wheels
export PROJECT_ROOT="$HOME/projects/varsplat-enhanced"
cp --no-preserve=all "$PROJECT_ROOT/cache/torch/hub/netvlad/VGG16-NetVLAD-Pitts30K.mat" cache/torch/hub/netvlad/
cp --no-preserve=all "$HOME/projects/varsplat-m0-environment-20261123/models/alexnet-owt-7be5be79.pth" cache/torch/hub/checkpoints/
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TMPDIR/mpl"
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$TMPDIR/cache"
export PIP_CACHE_DIR="$TMPDIR/pip-cache"
apptainer exec --nv --home "$TMPDIR/home" --bind "$TMPDIR:$TMPDIR" --bind "$PROJECT_ROOT:$PROJECT_ROOT:ro" /userhome/containers/jupyter.sif bash -c '
set -euo pipefail
source "$PROJECT_ROOT/scripts/env.sh"
cd "$TMPDIR/code"
python -m pip install --no-index --find-links "$TMPDIR/wheels" --find-links "$TMPDIR/eval_wheels" --no-build-isolation --no-deps --target "$TMPDIR/results/runtime" --report "$TMPDIR/results/install.json" \
  faiss-gpu==1.16.0 python-pycg==1.0.3 pyquaternion==0.9.9 omegaconf==2.3.0 \
  antlr4-python3-runtime==4.9.3 calmsize==0.1.3 screeninfo==0.8.1 evo==1.11.0 matplotlib==3.5.1
export TORCH_HOME="$TMPDIR/cache/torch"
export PYTHONPATH="$PWD:$TMPDIR/results/runtime:$PROJECT_ROOT/thirdparty/Hierarchical-Localization:$PROJECT_ROOT/thirdparty/evaluate_3d_reconstruction_lib"
python -B tools/m0/configure_plotting.py
python -B tools/m0/check_environment.py --output "$TMPDIR/results/environment.json"
python -B tools/m0/preflight.py runtime --output "$TMPDIR/results/runtime.json"
' > results/console.log 2>&1
