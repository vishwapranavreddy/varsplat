import copy
import unittest

from tools.analysis.aggregate_replays import aggregate_replays
from tools.analysis.fixed_frames import mean_metrics
from tools.m0.test_aggregate import reports


def fixtures():
    baselines = reports()
    replays = []
    for i, b in enumerate(baselines):
        b.update(job_id=str(i), expected_frames=3, rendered_frame_ids=list(range(i+1)),
                 pid=i+10, slam_started_unix=100, slam_finished_unix=200)
        per_frame = [dict(frame_id=f, psnr=20.+f, ssim=.8, lpips=.2,
                          depth_l1_train_view=.1) for f in range(3)]
        b['metrics']['rendering'] = mean_metrics(per_frame, b['rendered_frame_ids'])
        r = {k: b[k] for k in ('seed', 'dataset', 'scene', 'gpu', 'packages', 'input_content_sha256')}
        r.update(status='completed', baseline_sha256=str(i), baseline_job_id=str(i),
                 frame_ids=[0, 1, 2], per_frame=per_frame,
                 metrics=mean_metrics(per_frame, range(3)), script_sha256='frozen',
                 protocol='fixed observed frames', alexnet_sha256='pinned')
        replays.append(r)
    return baselines, ['0', '1', '2'], replays, [0, 1, 2]


class ReplayAggregateTests(unittest.TestCase):
    def test_different_native_coverage_same_full_protocol(self):
        args = fixtures()
        before = copy.deepcopy(args)
        result = aggregate_replays(*args)
        self.assertEqual(result['native_render_counts'], [1, 2, 3])
        self.assertEqual(result['metrics']['rendering_psnr'], {'mean': 21., 'std': 0.})
        self.assertFalse(result['native_rendering_aggregated'])
        self.assertEqual(args, before)

    def test_mispaired_and_changed_evaluator_rejected(self):
        for key in ('baseline_sha256', 'baseline_job_id', 'seed', 'input_content_sha256',
                    'script_sha256', 'alexnet_sha256'):
            args = fixtures()
            args[2][1][key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate_replays(*args)

    def test_missing_duplicate_or_fabricated_quality_rejected(self):
        for change in ('missing', 'duplicate', 'summary', 'native'):
            args = fixtures()
            r = args[2][1]
            if change == 'missing': r['frame_ids'].pop()
            if change == 'duplicate': r['per_frame'][1]['frame_id'] = 0
            if change == 'summary': r['metrics']['psnr'] += 1
            if change == 'native': args[0][1]['metrics']['rendering']['psnr'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate_replays(*args)

    def test_other_run_memory_rejected(self):
        args = fixtures()
        memories = [dict(report_sha256=str(i), pid=i+10, slam_started_unix=100,
                         slam_finished_unix=200, sampled_slam_max_process_bytes=1000+i)
                    for i in range(3)]
        result = aggregate_replays(*args, memories=memories)
        self.assertEqual(result['metrics']['sampled_slam_max_process_bytes']['mean'], 1001)
        memories[1]['pid'] = 999
        with self.assertRaises(ValueError):
            aggregate_replays(*args, memories=memories)


if __name__ == '__main__':
    unittest.main()
