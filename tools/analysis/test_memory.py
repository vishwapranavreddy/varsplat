from datetime import datetime, timezone
import unittest
from tools.analysis.memory import summarize


class MemoryTests(unittest.TestCase):
    def setUp(self):
        start = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc).timestamp()
        self.report = {'pid': 123, 'slam_complete': True, 'slam_started_unix': start, 'slam_finished_unix': start+1.5}
        self.samples = ('timestamp, gpu_uuid, pid, used_gpu_memory [MiB]\n'
                        '2026/11/23 17:30:00.000, GPU-a, 123, 100 MiB\n'
                        '2026/11/23 17:30:00.500, GPU-b, 456, 99999 MiB\n'
                        '2026/11/23 17:30:01.000, GPU-a, 123, 200 MiB\n'
                        '2026/11/23 17:30:02.000, GPU-a, 123, 99999 MiB\n')

    def test_excludes_other_job_and_offline_eval(self):
        r = summarize(self.report, self.samples, 19800)
        self.assertEqual(r['sampled_slam_max_process_bytes'], 200*1024**2)
        self.assertEqual(r['sample_count'], 2)

    def test_wrong_timezone_cannot_silently_return_peak(self):
        with self.assertRaises(ValueError): summarize(self.report, self.samples, 0)

    def test_missing_samples_rejected(self):
        self.report['slam_finished_unix'] += 10
        with self.assertRaises(ValueError): summarize(self.report, self.samples, 19800)


if __name__ == '__main__': unittest.main()
