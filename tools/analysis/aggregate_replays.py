"""Join verified full-frame replays to frozen runs before multi-seed aggregation.

Native rendering coverage may change with submap boundaries. This supplemental
protocol uses every associated frame and never rewrites the original reports.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools.analysis.fixed_frames import KEYS, mean_metrics
from tools.m0.aggregate import aggregate


def join_replay(baseline, baseline_sha256, replay):
    if (replay.get('status') != 'completed'
            or replay.get('baseline_sha256') != baseline_sha256
            or replay.get('baseline_job_id') != baseline['job_id']):
        raise ValueError('Replay is incomplete or belongs to another baseline')
    for key in ('seed', 'dataset', 'scene', 'gpu', 'packages', 'input_content_sha256'):
        if replay.get(key) != baseline[key]:
            raise ValueError('Replay identity mismatch: ' + key)
    ids = list(range(baseline['expected_frames']))
    if replay.get('frame_ids') != ids:
        raise ValueError('Replay must cover every associated frame in order')
    if len(replay['per_frame']) != len(ids):
        raise ValueError('Unexpected extra or missing replay frame')
    full = mean_metrics(replay['per_frame'], ids)
    native = mean_metrics(replay['per_frame'], baseline['rendered_frame_ids'])
    tolerances = {'psnr': .001, 'ssim': 1e-5, 'lpips': 1e-5, 'depth_l1_train_view': 1e-5}
    for label, actual, expected in (
            ('full', full, replay['metrics']),
            ('native', native, baseline['metrics']['rendering'])):
        if actual['num_renders'] != expected['num_renders']:
            raise ValueError('Incorrect rendering denominator: ' + label)
        for key in KEYS:
            limit = 1e-10 if label == 'full' else tolerances[key]
            if not math.isfinite(expected[key]) or abs(actual[key]-expected[key]) > limit:
                raise ValueError('Rendering summary or native parity failed: ' + label + '/' + key)
    joined = copy.deepcopy(baseline)
    joined['rendered_frame_ids'] = ids
    joined['metrics']['rendering'] = full
    return joined


def aggregate_replays(baselines, hashes, replays, seeds, memories=None):
    if not len(baselines) == len(hashes) == len(replays):
        raise ValueError('One replay and hash required for every baseline')
    # Require identical evaluator, learned metric weights and declared protocol.
    identities = {
        json.dumps({k: r[k] for k in ('script_sha256', 'protocol', 'alexnet_sha256')}, sort_keys=True)
        for r in replays}
    if len(identities) != 1:
        raise ValueError('Replay evaluator or metric weights differ')
    joined = [join_replay(b, h, r) for b, h, r in zip(baselines, hashes, replays)]
    result = aggregate(joined, seeds)
    result['rendering_protocol'] = json.loads(next(iter(identities)))
    result['native_rendering_aggregated'] = False
    result['baseline_reports_sha256'] = hashes
    result['native_render_counts'] = [len(b['rendered_frame_ids']) for b in baselines]
    if memories is not None:
        if len(memories) != len(baselines):
            raise ValueError('One memory summary required per baseline')
        values = []
        for b, h, m in zip(baselines, hashes, memories):
            if (m.get('report_sha256') != h or m.get('pid') != b['pid']
                    or m.get('slam_started_unix') != b['slam_started_unix']
                    or m.get('slam_finished_unix') != b['slam_finished_unix']):
                raise ValueError('Memory summary belongs to another process/run')
            value = m['sampled_slam_max_process_bytes']
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid measured memory')
            values.append(value)
        result['metrics']['sampled_slam_max_process_bytes'] = {
            'mean': statistics.mean(values), 'std': statistics.stdev(values)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baselines', nargs='+', type=Path, required=True)
    p.add_argument('--replays', nargs='+', type=Path, required=True)
    p.add_argument('--memories', nargs='+', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    read = lambda paths: [json.loads(path.read_bytes()) for path in paths]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.baselines]
    result = aggregate_replays(read(args.baselines), hashes, read(args.replays), args.seeds,
                               read(args.memories) if args.memories else None)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
