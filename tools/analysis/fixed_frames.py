"""Supplemental full-sequence rendering audit; does not replace upstream metrics.

Reload the completed upstream global PLY, evaluate every associated input frame,
and verify that its exported-keyframe subset reproduces upstream quality metrics.
Run on an allocated GPU, after SLAM/evaluation timing has finished.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.m0.run import sha256, package_versions, source_manifest

KEYS = ('psnr', 'ssim', 'lpips', 'depth_l1_train_view')


def mean_metrics(rows, frame_ids):
    ids = list(frame_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Require a nonempty, unique frame set')
    indexed = {r['frame_id']: r for r in rows}
    if len(indexed) != len(rows) or not set(ids).issubset(indexed):
        raise ValueError('Missing or duplicate measured frames')
    selected = [indexed[i] for i in ids]
    if any(not math.isfinite(r[k]) for r in selected for k in KEYS):
        raise ValueError('Non-finite quality measurement')
    return {**{k: sum(r[k] for r in selected) / len(ids) for k in KEYS}, 'num_renders': len(ids)}


def evaluate(run_dir, input_dir, result):
    import numpy as np
    import torch
    from pytorch_msssim import ms_ssim
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from src.entities.datasets import get_dataset
    from src.entities.gaussian_model import GaussianModel
    from src.utils.utils import get_render_settings, render_gaussian_model, setup_seed

    baseline_path = run_dir / 'benchmark.json'
    baseline = json.loads(baseline_path.read_text())
    if baseline['status'] != 'completed' or not baseline.get('slam_complete') or baseline['evaluation_mode'] != 'upstream':
        raise ValueError('Require a completed upstream baseline including quality evaluation')
    if baseline['dataset'] not in ('tum_rgbd', 'replica', 'scan_net'):
        raise ValueError('This observed-view protocol does not implement ScanNet++ novel views')
    result.update(baseline_sha256=sha256(baseline_path), baseline_job_id=baseline['job_id'],
                  seed=baseline['seed'], dataset=baseline['dataset'], scene=baseline['scene'],
                  gpu=torch.cuda.get_device_name(), packages=package_versions())
    if result['packages'] != baseline['packages'] or result['gpu'] != baseline['gpu']:
        raise ValueError('Use the baseline package overlay and GPU model for replay')
    current_sources = source_manifest()
    for name, digest in baseline['source_manifest'].items():
        if name.startswith(('src/', 'configs/')) and current_sources.get(name) != digest:
            raise ValueError('Upstream source/config changed: ' + name)
    import gaussian_rasterizer
    if sha256(gaussian_rasterizer._C.__file__) != baseline['native_sha256']['gaussian_rasterizer']:
        raise ValueError('Mapping rasterizer binary changed')
    cfg = json.loads(json.dumps(baseline['config']))
    cfg['data']['input_path'] = str(input_dir.resolve())
    dataset = get_dataset(cfg['dataset_name'])({**cfg['data'], **cfg['cam']})
    if len(dataset) != baseline['expected_frames']:
        raise ValueError('Input frame count changed')
    content = hashlib.sha256()
    for i, association in enumerate(baseline['frame_associations']):
        if association['id'] != i:
            raise ValueError('Noncontiguous baseline association IDs')
        for kind, paths in (('rgb', dataset.color_paths), ('depth', dataset.depth_paths)):
            digest = sha256(paths[i])
            if digest != association[kind + '_sha256']:
                raise ValueError(f'Input mismatch: {kind} frame {i}')
            content.update(bytes.fromhex(digest))
    if len(baseline['frame_associations']) != len(dataset):
        raise ValueError('Incomplete association manifest')
    gt = np.asarray(dataset.poses[:len(dataset)], dtype=np.float64)
    content.update(gt.astype('<f8').tobytes())
    if content.hexdigest() != baseline['input_content_sha256']:
        raise ValueError('Input content or pose association changed')
    result['input_content_sha256'] = content.hexdigest()
    poses_path = run_dir / 'estimated_c2w.ckpt'
    poses = torch.load(poses_path, map_location='cpu', weights_only=False).numpy()
    from tools.m0.metrics import ate_se3
    # Checks full count, finite values and SE(3); never drops bad poses.
    if not np.isclose(ate_se3(poses, gt)['rmse_m'], baseline['metrics']['ate_all_frames']['rmse_m'], rtol=1e-7, atol=1e-9):
        raise ValueError('Saved trajectory does not match baseline report')
    result['trajectory_sha256'] = sha256(poses_path)
    setup_seed(baseline['seed'])
    ply = run_dir / (baseline['scene'] + '_global_splats.ply')
    result['global_ply_sha256'] = sha256(ply)
    model = GaussianModel(3)
    model.load_ply(str(ply))
    alexnet = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
    if sha256(alexnet) != '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02':
        raise ValueError('Missing or mismatched offline AlexNet weights')
    result['alexnet_sha256'] = sha256(alexnet)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda().eval()
    rows = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(len(dataset)):
            _, color, depth, _ = dataset[i]
            target = torch.from_numpy(color.copy()).permute(2, 0, 1).float().cuda() / 255.
            depth = torch.from_numpy(depth.copy()).cuda()
            settings = get_render_settings(dataset.width, dataset.height, dataset.intrinsics, np.linalg.inv(poses[i]))
            render = render_gaussian_model(model, settings)
            rgb = render['color'].clamp(0., 1.)
            mse = torch.nn.functional.mse_loss(rgb, target)
            row = {'frame_id': i, 'psnr': (-10. * torch.log10(mse)).item(),
                         'ssim': ms_ssim(rgb[None], target[None], data_range=1., size_average=True).item(),
                         'lpips': lpips(rgb[None], target[None]).item(),
                   'depth_l1_train_view': (render['depth'][0] - depth).abs().mean().item()}
            if any(not math.isfinite(row[k]) for k in KEYS):
                raise ValueError(f'Non-finite rendering metric at frame {i}')
            rows.append(row)
    torch.cuda.synchronize()
    result['render_metric_seconds'] = time.perf_counter() - start
    result['frame_ids'] = list(range(len(dataset)))
    result['per_frame'] = rows
    result['metrics'] = mean_metrics(rows, result['frame_ids'])
    result['upstream_subset_replay'] = mean_metrics(rows, baseline['rendered_frame_ids'])
    native = baseline['metrics']['rendering']
    replay = result['upstream_subset_replay']
    result['upstream_subset_absolute_errors'] = {k: abs(native[k] - replay[k]) for k in KEYS}
    result['parity_absolute_tolerances'] = {'psnr': .001, 'ssim': 1e-5, 'lpips': 1e-5, 'depth_l1_train_view': 1e-5}
    if native['num_renders'] != replay['num_renders'] or any(result['upstream_subset_absolute_errors'][k] > result['parity_absolute_tolerances'][k] for k in KEYS):
        raise ValueError('Reloaded PLY does not reproduce native-subset metrics within declared tolerances')
    result['status'] = 'completed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the repository root')
    # Reserve output before GPU work; never overwrite a prior report.
    with args.output.open('x') as f:
        result = {'schema_version': 1, 'status': 'started', 'script_sha256': sha256(__file__),
                  'protocol': 'all associated observed frames; saved upstream refined global PLY; estimated poses',
                  'role': 'supplemental coverage audit, not upstream paper protocol or unseen-view generalization'}
        json.dump(result, f)
    start = time.perf_counter()
    try:
        evaluate(args.run_dir.resolve(), args.input, result)
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    result['total_seconds'] = time.perf_counter() - start
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_frame', 'packages', 'frame_ids')}))
    raise SystemExit(0 if result['status'] == 'completed' else 1)
