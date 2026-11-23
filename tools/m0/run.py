"""Baseline-preserving runner. Original src/, entrypoints and scene configs are untouched.

Run from the repository root inside its existing CUDA environment.
--profile stages is diagnostic only: boundary synchronizations perturb runtime.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_manifest():
    paths = [ROOT / n for n in ('run_slam.py', 'run_evaluation.py', 'requirements.txt', '.gitmodules')]
    for folder, pattern in (('src', '*.py'), ('configs', '*.yaml'), ('tools/m0', '*.py')):
        paths.extend((ROOT / folder).rglob(pattern))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def package_versions():
    # First distribution wins, matching import path precedence for overlays.
    versions = {}
    for d in importlib.metadata.distributions():
        if d.metadata['Name']:
            versions.setdefault(d.metadata['Name'], d.version)
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--profile', choices=['wall', 'stages'], default=None)
    parser.add_argument('--evaluation', choices=['upstream', 'raw', 'none'], default='upstream')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the repository root (upstream config inheritance is CWD-relative)')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'started', 'seed': args.seed,
              'evaluation_mode': args.evaluation, 'source_manifest': source_manifest(),
              'job_id': os.environ.get('SLURM_JOB_ID'), 'pid': os.getpid(), 'python': platform.python_version(),
              'config_flavor': os.environ.get('M0_CONFIG_FLAVOR', 'unspecified'),
              'argv': sys.argv, 'phases': {}, 'metrics': {}, 'failures': []}
    def save():
        temp = out / 'benchmark.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temp.replace(out / 'benchmark.json')
    save()
    process_start = time.perf_counter()
    profiler = None
    try:
        import numpy as np
        import torch
        from src.entities.gaussian_slam import GaussianSLAM
        from src.evaluation.evaluator import Evaluator
        from src.utils.io_utils import load_config
        from src.utils.utils import setup_seed
        from tools.m0.metrics import ate_se3
        from tools.m0.profiler import Profiler, instrument_stages
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA allocation is required')
        cfg = load_config(args.config)
        cfg['data'].update(input_path=str(Path(args.input).resolve()), output_path=str(out))
        cfg['seed'] = args.seed  # Includes seed 0, unlike the upstream truthiness CLI check.
        cfg['use_wandb'] = False
        profile = args.profile or cfg.get('benchmark', {}).get('profile', 'wall')
        if profile not in ('wall', 'stages'):
            raise ValueError('Unknown benchmark.profile')
        report.update(profile=profile, config=cfg, dataset=cfg['dataset_name'], scene=cfg['data']['scene_name'],
                      gpu=torch.cuda.get_device_name(), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                      cuda_build=torch.version.cuda, torch=torch.__version__,
                      packages=package_versions())
        import gaussian_rasterizer, diff_gaussian_rasterization, faiss
        report['native_sha256'] = {m.__name__: sha256(m._C.__file__) for m in (gaussian_rasterizer, diff_gaussian_rasterization)}
        report['faiss_loaded'] = {'path': faiss.__file__, 'version': faiss.__version__}
        # No downloads on compute nodes: hloc otherwise downloads silently on a miss.
        weight = Path(torch.hub.get_dir()) / 'netvlad/VGG16-NetVLAD-Pitts30K.mat'
        if not weight.is_file():
            raise FileNotFoundError('Stage the official NetVLAD checkpoint offline before this run: ' + str(weight))
        report['netvlad_sha256'] = sha256(weight)
        setup_seed(args.seed)
        profiler = Profiler(torch.cuda.synchronize)
        torch.cuda.reset_peak_memory_stats()
        with profiler.span('initialization'):
            slam = GaussianSLAM(cfg)
        report['initialization_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['expected_frames'] = len(slam.dataset)
        # Store association identity independently of paths altered by scratch staging.
        report['frame_associations'] = [
            {'id': i, 'rgb': str(slam.dataset.color_paths[i]), 'depth': str(slam.dataset.depth_paths[i])}
            for i in range(len(slam.dataset))]
        # Outside timed SLAM: compare bytes, not merely equal-looking filenames.
        input_start = time.perf_counter()
        content = hashlib.sha256()
        for association in report['frame_associations']:
            for kind in ('rgb', 'depth'):
                value = sha256(association[kind])
                association[kind + '_sha256'] = value
                content.update(bytes.fromhex(value))
        gt = np.asarray(slam.dataset.poses[:len(slam.dataset)], dtype=np.float64)
        content.update(gt.astype('<f8').tobytes())
        report['input_content_sha256'] = content.hexdigest()
        report['input_hash_seconds'] = time.perf_counter() - input_start
        save()
        if profile == 'stages':
            instrument_stages(slam, profiler)
        report['slam_started_unix'] = time.time()
        save()
        with profiler.span('slam'):
            slam.run()
        report['slam_finished_unix'] = time.time()
        report['slam_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['slam_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        pred = slam.estimated_c2ws.detach().cpu().numpy()
        report['metrics']['ate_all_frames'] = ate_se3(pred, gt)
        report['metrics']['ate_mapping_keyframes'] = ate_se3(pred, gt, sorted(set(slam.mapping_frame_ids)))
        exported = set()
        for path in sorted((out / 'submaps').glob('*.ckpt')):
            exported.update(torch.load(path, map_location='cpu', weights_only=False)['submap_keyframes'])
        report['exported_keyframe_count'] = len(exported)
        report['mapping_keyframe_count'] = len(set(slam.mapping_frame_ids))
        if len(exported) >= 3:
            report['metrics']['ate_exported_keyframes'] = ate_se3(pred, gt, sorted(exported))
        report['slam_complete'] = True
        save()
        if args.evaluation != 'none':
            torch.cuda.reset_peak_memory_stats()
            with profiler.span('evaluation'):
                evaluator = Evaluator(out, out / 'config.yaml')
                if args.evaluation == 'upstream':
                    evaluator.run()
                else:
                    evaluator.run_rendering_eval()
            report['evaluation_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            metric_file = out / ('rendering_metrics_global.json' if args.evaluation == 'upstream' else 'rendering_metrics.json')
            if cfg['dataset_name'] == 'scannetpp' and args.evaluation == 'upstream':
                metric_file = out / 'nvs_eval/results.json'
            if not metric_file.is_file():
                raise RuntimeError('Evaluator did not emit required quality metrics: ' + str(metric_file))
            report['metrics']['rendering'] = json.loads(metric_file.read_text())
            if cfg['dataset_name'] != 'scannetpp':
                report['rendered_frame_ids'] = sorted(exported)
                if report['metrics']['rendering'].get('num_renders') != len(exported):
                    raise ValueError('Rendering coverage does not match exported keyframe IDs')
            if any(v is None or not np.isfinite(v) for v in report['metrics']['rendering'].values()):
                raise ValueError('Missing/non-finite rendering metric')
        report['status'] = 'completed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['failures'].append({'exception': repr(exc), 'traceback': traceback.format_exc()})
    finally:
        if profiler is not None:
            report['timings'] = profiler.rows
            slam_rows = [r for r in profiler.rows if r['stage'] == 'slam' and r['status'] == 'completed']
            if slam_rows:
                report['slam_wall_seconds'] = slam_rows[0]['seconds']
                report['slam_seconds_per_input_frame'] = slam_rows[0]['seconds'] / report['expected_frames']
        report['runner_wall_seconds'] = time.perf_counter() - process_start
        save()
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
