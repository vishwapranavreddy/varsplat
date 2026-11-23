"""Small allocated-GPU checks; never installs dependencies or downloads inputs.

Modes are independent: runtime audits the installed environment and author cache;
variance checks each rendered variance channel against finite differences.
These are diagnostic results, not a baseline reproduction.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def runtime():
    import torch
    import numpy as np
    import faiss
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "numpy": np.__version__,
              "gpu": torch.cuda.get_device_name(),
              "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "faiss_gpu_api": all(hasattr(faiss, k) for k in ('StandardGpuResources', 'index_cpu_to_gpu', 'index_cpu_to_all_gpus')),
              "packages": {}, "modules": {}, "blockers": []}
    for name in ('open3d', 'faiss-cpu', 'faiss-gpu', 'numpy', 'torchvision', 'roma', 'pytorch-msssim', 'torchmetrics'):
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    for name in ('gaussian_rasterizer', 'diff_gaussian_rasterization', 'simple_knn._C', 'src.entities.gaussian_slam', 'src.evaluation.evaluator'):
        try:
            mod = importlib.import_module(name)
            report['modules'][name] = {"path": mod.__file__, "sha256": digest(mod.__file__)}
            native = getattr(mod, '_C', None)
            if native is not None:
                report['modules'][name]['native_sha256'] = digest(native.__file__)
        except Exception as exc:
            report['modules'][name] = {"error": repr(exc)}
            report['blockers'].append(name + ': ' + repr(exc))
    if not report['faiss_gpu_api']:
        report['blockers'].append('Upstream mapping/overlap require GPU FAISS; CPU fallback changes baseline environment')
    weight = Path(torch.hub.get_dir()) / 'netvlad/VGG16-NetVLAD-Pitts30K.mat'
    report['netvlad_weight'] = {"path": str(weight), "exists": weight.is_file()}
    if weight.is_file():
        report['netvlad_weight']['sha256'] = digest(weight)
    else:
        report['blockers'].append('Missing offline NetVLAD weight')
    old = Path(os.environ['PROJECT_ROOT'])
    author = old / 'baselines/author_fr1_desk'
    report['author_checkpoint'] = {}
    for name in ('config.yaml', 'estimated_c2w.ckpt'):
        p = author / name
        report['author_checkpoint'][name] = {"exists": p.is_file(), "sha256": digest(p) if p.is_file() else None}
    if (author / 'estimated_c2w.ckpt').is_file():
        pred = torch.load(author / 'estimated_c2w.ckpt', map_location='cpu', weights_only=False)
        report['author_checkpoint']['pose_count'] = len(pred)
        report['author_checkpoint']['finite'] = bool(torch.isfinite(pred).all())
    # Import/load only; no model training or modification of author checkpoints.
    try:
        from src.entities.datasets import TUM_RGBD
        from src.utils.io_utils import load_config
        from tools.m0.metrics import ate_se3
        cfg = load_config(str(ROOT / 'configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml'))
        cfg['data']['input_path'] = str(old / 'data/rgbd_dataset_freiburg1_desk')
        dataset = TUM_RGBD({**cfg['data'], **cfg['cam']})
        report['dataset_frames'] = len(dataset)
        report['author_ate_all_frames'] = ate_se3(pred.numpy(), np.asarray(dataset.poses))
        kfs = set()
        for p in sorted((author / 'submaps').glob('*.ckpt')):
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            kfs.update(ckpt['submap_keyframes'])
        report['author_ate_exported_keyframes'] = ate_se3(pred.numpy(), np.asarray(dataset.poses), sorted(kfs))
        report['author_submap_keyframe_count'] = len(kfs)
    except Exception as exc:
        report['author_evaluation_error'] = repr(exc)
    report['status'] = 'BLOCKED' if report['blockers'] else 'PREFLIGHT_PASSED'
    return report


def variance():
    import numpy as np
    import torch
    from src.entities.gaussian_model import GaussianModel
    from src.utils.utils import get_render_settings, render_gaussian_model
    # A single splat, black appearance, unsaturated variance, deterministic geometry.
    model = GaussianModel(0)
    values = {'_xyz': [[0., 0., 2.]], '_features_dc': [[[-1.77245385] * 3]],
              '_scaling': [[-1.5] * 3], '_rotation': [[1., 0., 0., 0.]],
              '_opacity': [[0.]], '_color_variance': [[-4., -3.5, -3.]]}
    for name, value in values.items():
        setattr(model, name, torch.nn.Parameter(torch.tensor(value, device='cuda', dtype=torch.float32)))
    model._features_rest = torch.nn.Parameter(torch.empty(1, 0, 3, device='cuda'))
    K = np.array([[30., 0, 16], [0, 30., 16], [0, 0, 1.]])
    settings = get_render_settings(32, 32, K, np.eye(4))
    rows = []
    for channel in range(3):
        for parameter in model.__dict__.values():
            if isinstance(parameter, torch.nn.Parameter):
                parameter.grad = None
        value = render_gaussian_model(model, settings)['variance'][channel, 16, 16]
        value.backward()
        analytic = model._color_variance.grad.detach().cpu().numpy()[0].copy()
        finite = []
        step = 1e-2
        for j in range(3):
            original = model._color_variance[0, j].item()
            with torch.no_grad():
                model._color_variance[0, j] = original + step
                plus = render_gaussian_model(model, settings)['variance'][channel, 16, 16].item()
                model._color_variance[0, j] = original - step
                minus = render_gaussian_model(model, settings)['variance'][channel, 16, 16].item()
                model._color_variance[0, j] = original
            finite.append((plus - minus) / (2 * step))
        rows.append({"output_channel": channel, "variance": value.item(), "autograd": analytic.tolist(),
                     "finite_difference": finite, "matches": bool(np.allclose(analytic, finite, rtol=0.02, atol=1e-5))})
    torch.cuda.synchronize()
    return {"status": 'GRADIENT_CHECK_PASSED' if all(r['matches'] for r in rows) else 'GRADIENT_MISMATCH',
            "rows": rows, "scope": 'installed mapping rasterizer; original kernel unchanged',
            "gpu": torch.cuda.get_device_name()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['runtime', 'variance'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = runtime() if args.mode == 'runtime' else variance()
    except Exception as exc:
        result = {"status": 'ERROR', "error": repr(exc), "traceback": traceback.format_exc()}
    result.update(mode=args.mode, job_id=os.environ.get('SLURM_JOB_ID'))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'modules', 'author_ate_all_frames', 'author_ate_exported_keyframes')}))
    # Diagnostic execution success is distinct from scientific checks passing.
    raise SystemExit(1 if result['status'] == 'ERROR' else 0)
