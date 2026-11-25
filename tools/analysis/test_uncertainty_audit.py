import unittest
import numpy as np

from tools.analysis.uncertainty_audit import calibration_statistics, centered_weights, scale_sensitivity


class UncertaintyAuditTests(unittest.TestCase):
    def test_global_scale_cancels_but_spatial_scale_does_not(self):
        v = np.array([[.001, .002, .003], [.005, .006, .007], [.01, .02, .03]])
        mask = np.ones(3, dtype=bool)
        report = scale_sensitivity(v, mask, 2., 3.)
        for row in report['scale_tests']:
            self.assertLess(row['max_weight_change_without_epsilon'], 1e-14)
        altered = v.copy(); altered[0] *= 2
        self.assertGreater(np.max(np.abs(centered_weights(v, mask, 2, 3, 0)-centered_weights(altered, mask, 2, 3, 0))), .01)

    def test_even_count_uses_lower_median_as_torch_does(self):
        v = np.tile(np.array([1., 4., 9., 16.])[:, None], (1, 3))
        w = centered_weights(v, np.ones(4, dtype=bool), 1, 100, 0)
        np.testing.assert_allclose(w[:, 0], [4., 1., 4/9, .25])

    def test_likelihood_and_coverage_known_values(self):
        v = np.ones((3, 3))
        r = np.tile(np.array([0., 1., 3.])[:, None], (1, 3))
        result = calibration_statistics(r, v, np.ones(3, dtype=bool))
        np.testing.assert_allclose(result['standardized_squared_residual_by_channel'], [10/3]*3)
        np.testing.assert_allclose(result['one_sigma_coverage_by_channel'], [2/3]*3)
        self.assertEqual(sum(b['count'] for b in result['variance_bins_by_channel'][0]), 3)

    def test_invalid_selected_data_and_empty_masks_fail(self):
        for v, mask in [(np.ones((2, 3)), np.zeros(2, dtype=bool)),
                        (np.zeros((2, 3)), np.ones(2, dtype=bool)),
                        (np.full((2, 3), np.nan), np.ones(2, dtype=bool))]:
            with self.assertRaises(ValueError): centered_weights(v, mask, 1, 3)

    def test_invalid_unselected_data_is_excluded(self):
        v = np.array([[1., 1., 1.], [np.nan]*3])
        r = np.zeros((2, 3)); r[1] = np.nan
        result = calibration_statistics(r, v, np.array([True, False]))
        self.assertEqual(result['valid_observations'], 1)


if __name__ == '__main__':
    unittest.main()
