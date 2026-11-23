import copy
import unittest
from tools.m0.aggregate import aggregate
from tools.m0.profiler import Profiler


def reports():
    rows = []
    for seed in range(3):
        r = dict(seed=seed, status='completed', slam_complete=True, profile='wall',
                 config={'seed': seed, 'data': {'input_path': 'scratch', 'output_path': str(seed)}, 'tracking': {'iterations': 200}},
                 frame_associations=[{'rgb': '/scratch/rgb/a.png', 'depth': '/scratch/depth/a.png'}],
                 metrics={'ate_all_frames': {'rmse_m': .01 * (seed+1)}, 'ate_mapping_keyframes': {'rmse_m': .01}},
                 slam_wall_seconds=100+seed, slam_seconds_per_input_frame=1+seed/100,
                 slam_peak_allocated_bytes=1000, slam_peak_reserved_bytes=2000)
        for key in ('dataset', 'scene', 'gpu', 'gpu_bytes', 'torch', 'cuda_build', 'packages', 'source_manifest', 'native_sha256', 'netvlad_sha256', 'input_content_sha256', 'evaluation_mode', 'expected_frames'):
            r[key] = 'same'
        rows.append(r)
    return rows


class AggregateTests(unittest.TestCase):
    def test_mean_and_sample_std(self):
        r = aggregate(reports(), [0, 1, 2])
        self.assertAlmostEqual(r['metrics']['ate_all_frames_m']['mean'], .02)
        self.assertAlmostEqual(r['metrics']['ate_all_frames_m']['std'], .01)

    def test_reject_missing_duplicate_failed_and_profiled(self):
        for change in ('missing', 'duplicate', 'failed', 'profiled'):
            rows = reports()
            if change == 'missing': rows.pop()
            if change == 'duplicate': rows[2]['seed'] = 1
            if change == 'failed': rows[0]['status'] = 'failed'
            if change == 'profiled': rows[0]['profile'] = 'stages'
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate(rows, [0, 1, 2])

    def test_reject_mismatched_inputs_config_and_coverage(self):
        for change in ('bytes', 'config', 'render'):
            rows = reports()
            if change == 'bytes': rows[1]['input_content_sha256'] = 'different'
            if change == 'config': rows[1]['config']['tracking']['iterations'] = 100
            if change == 'render': rows[1]['metrics']['rendering'] = {'psnr': 30}
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate(rows, [0, 1, 2])

    def test_reject_nonfinite(self):
        rows = reports()
        rows[1]['slam_wall_seconds'] = float('nan')
        with self.assertRaises(ValueError): aggregate(rows, [0, 1, 2])

    def test_profiler_preserves_failures_and_nesting(self):
        calls = []
        p = Profiler(lambda: calls.append(1))
        with self.assertRaisesRegex(RuntimeError, 'expected'):
            with p.span('outer'):
                with p.span('inner'):
                    raise RuntimeError('expected')
        self.assertEqual([r['nesting_depth'] for r in p.rows], [1, 0])
        self.assertTrue(all(r['status'] == 'failed' for r in p.rows))
        self.assertEqual(len(calls), 4)
        self.assertEqual(p.depth, 0)


if __name__ == '__main__': unittest.main()
