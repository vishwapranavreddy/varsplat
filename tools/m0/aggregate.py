"""Aggregate only complete, comparable repetitions; never infer missing measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def aggregate(reports, expected_seeds):
    if len(expected_seeds) < 3 or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError('Specify at least three distinct seeds')
    if sorted(r['seed'] for r in reports) != sorted(expected_seeds):
        raise ValueError('Missing, extra, or duplicate seed')
    signatures = set()
    rows = []
    for r in reports:
        if r['status'] != 'completed' or not r.get('slam_complete') or r['profile'] != 'wall':
            raise ValueError('Only successful wall-mode full SLAM runs are comparable')
        config = json.loads(json.dumps(r['config']))
        config.pop('seed', None)
        config['data'].pop('output_path', None)
        config['data'].pop('input_path', None)
        signature = {k: r[k] for k in ('dataset', 'scene', 'gpu', 'gpu_bytes', 'torch', 'cuda_build', 'packages', 'source_manifest', 'native_sha256', 'netvlad_sha256', 'input_content_sha256', 'evaluation_mode', 'expected_frames')}
        signature['config'] = config
        signature['rendered_frame_ids'] = r.get('rendered_frame_ids')
        signature['association_names'] = [(Path(x['rgb']).name, Path(x['depth']).name) for x in r['frame_associations']]
        signatures.add(hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest())
        row = {'ate_all_frames_m': r['metrics']['ate_all_frames']['rmse_m'],
               'ate_mapping_keyframes_m': r['metrics']['ate_mapping_keyframes']['rmse_m'],
               'slam_wall_seconds': r['slam_wall_seconds'],
               'slam_seconds_per_input_frame': r['slam_seconds_per_input_frame'],
               'slam_peak_allocated_bytes': r['slam_peak_allocated_bytes'],
               'slam_peak_reserved_bytes': r['slam_peak_reserved_bytes']}
        for key, value in r['metrics'].get('rendering', {}).items():
            if key != 'num_renders':
                row['rendering_' + key] = value
        rows.append(row)
    if len(signatures) != 1 or any(set(row) != set(rows[0]) for row in rows):
        raise ValueError('Mismatched source, config, environment, association or metric coverage')
    result = {'seeds': sorted(expected_seeds), 'n': len(reports), 'std_ddof': 1,
              'comparison_signature': next(iter(signatures)), 'metrics': {}}
    for key in rows[0]:
        values = [row[key] for row in rows]
        import math
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('Non-finite metric: ' + key)
        result['metrics'][key] = {'mean': statistics.mean(values), 'std': statistics.stdev(values)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--expected-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = aggregate([json.loads(p.read_text()) for p in args.reports], args.expected_seeds)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
