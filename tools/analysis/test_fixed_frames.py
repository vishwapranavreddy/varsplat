import unittest
from tools.analysis.fixed_frames import KEYS, mean_metrics


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(frame_id=i, **{k: value for k in KEYS})
                     for i, value in [(0, 10.), (1, 20.), (2, 90.)]]

    def test_unmeasured_tail_cannot_hide_in_mean(self):
        self.assertEqual(mean_metrics(self.rows, [0, 1])['psnr'], 15.)
        self.assertEqual(mean_metrics(self.rows, [0, 1, 2])['psnr'], 40.)
        with self.assertRaises(ValueError): mean_metrics(self.rows[:2], [0, 1, 2])

    def test_invalid_coverage_and_values_fail(self):
        for ids in ([], [0, 0], [0, 3]):
            with self.assertRaises(ValueError): mean_metrics(self.rows, ids)
        with self.assertRaises(ValueError): mean_metrics(self.rows + [self.rows[0]], [0])
        self.rows[2]['psnr'] = float('nan')
        with self.assertRaises(ValueError): mean_metrics(self.rows, [0, 1, 2])


if __name__ == '__main__': unittest.main()
