"""Exercise upstream FAISS GPU operations without a CPU fallback."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import traceback


def check():
    import torch
    import numpy as np
    import faiss
    import faiss.contrib.torch_utils
    from src.entities.gaussian_slam import GaussianSLAM
    from src.evaluation.evaluator import Evaluator
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
    import gaussian_rasterizer
    native = Path(gaussian_rasterizer._C.__file__)
    return {'status': 'PASSED', 'faiss': faiss.__version__, 'faiss_path': faiss.__file__,
            'gpu': torch.cuda.get_device_name(), 'numpy': np.__version__,
            'torch': torch.__version__, 'mapping_native_sha256': hashlib.sha256(native.read_bytes()).hexdigest(),
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
            'checks': ['upstream_imports', 'numpy_gpu_l2_vs_reference', 'torch_gpu_l2_vs_reference']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    try:
        report = check()
    except Exception as exc:
        report = {'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc()}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'packages'}))
    raise SystemExit(0 if report['status'] == 'PASSED' else 1)
