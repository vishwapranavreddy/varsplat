"""Exercise upstream FAISS GPU operations without a CPU fallback."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import traceback


def check(output_dir):
    import torch
    import numpy as np
    import faiss
    import faiss.contrib.torch_utils
    from src.entities.gaussian_slam import GaussianSLAM
    from src.evaluation.evaluator import Evaluator
    import evo, matplotlib
    evo_version = importlib.metadata.version('evo')
    if evo_version != '1.11.0':
        raise RuntimeError('Upstream requires evo==1.11.0, loaded ' + evo_version)
    if matplotlib.__version__ != '3.5.1':
        raise RuntimeError('Upstream pins matplotlib==3.5.1, loaded ' + matplotlib.__version__)
    rng = np.random.default_rng(123)
    points = rng.normal(size=(200, 3)).astype('float32')
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(3))
    index.add(points)
    d, ids = index.search(points[:10], 4)
    reference = ((points[:10, None] - points[None]) ** 2).sum(-1)
    expected = np.argsort(reference, axis=1)[:, :4]
    np.testing.assert_array_equal(ids, expected)
    np.testing.assert_allclose(d, np.take_along_axis(reference, expected, axis=1), atol=2e-5)
    index2 = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(3))
    index2.add(torch.from_numpy(points).cuda())
    dt, it = index2.search(torch.from_numpy(points[:10]).cuda(), 4)
    np.testing.assert_array_equal(it.cpu().numpy(), expected)
    torch.cuda.synchronize()
    # Imports alone missed the removed evo API reached at the first PGO event.
    # Exercise the actual upstream alignment, metric, colormap and plot export.
    from src.utils.eval_utils import evaluate_evo
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 1, 0]]
    estimates = poses.copy()
    estimates[:, :3, 3] += rng.normal(scale=.01, size=(6, 3))
    plot_dir = output_dir / 'evo_smoke'
    plot_dir.mkdir(exist_ok=True)
    diagnostic_ate = evaluate_evo(list(poses), list(estimates), str(plot_dir), 'smoke')
    if not np.isfinite(diagnostic_ate): raise ValueError('Non-finite evo diagnostic')
    from matplotlib import pyplot as plt
    plt.close('all')
    from tools.m0.smoke_quality import check as check_quality
    quality_smoke = check_quality()
    import gaussian_rasterizer
    native = Path(gaussian_rasterizer._C.__file__)
    versions = {}
    for d in importlib.metadata.distributions():
        if d.metadata['Name']: versions.setdefault(d.metadata['Name'], d.version)
    return {'status': 'PASSED', 'faiss': faiss.__version__, 'faiss_path': faiss.__file__,
            'gpu': torch.cuda.get_device_name(), 'numpy': np.__version__,
            'torch': torch.__version__, 'mapping_native_sha256': hashlib.sha256(native.read_bytes()).hexdigest(),
            'evo': evo_version, 'evo_path': evo.__file__, 'matplotlib': matplotlib.__version__,
            'quality_smoke': quality_smoke,
            'packages': versions,
            'checks': ['upstream_imports', 'numpy_gpu_l2_vs_reference', 'torch_gpu_l2_vs_reference', 'upstream_evo_alignment_and_plot']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    try:
        report = check(args.output.parent)
    except Exception as exc:
        report = {'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc()}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'packages'}))
    raise SystemExit(0 if report['status'] == 'PASSED' else 1)
