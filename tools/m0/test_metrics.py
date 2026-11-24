import ast
import inspect
from pathlib import Path
import unittest
import numpy as np
from tools.m0.metrics import ate_se3


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.repeat(np.eye(4)[None], 5, axis=0)
        self.reference[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]]

    def test_rigid_alignment(self):
        transform = np.eye(4)
        transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        transform[:3, 3] = [4, 5, 6]
        self.assertLess(ate_se3(transform @ self.reference, self.reference)['rmse_m'], 1e-12)

    def test_scale_not_fitted(self):
        pred = self.reference.copy()
        pred[:, :3, 3] *= 2
        self.assertGreater(ate_se3(pred, self.reference)['rmse_m'], 0.5)

    def test_missing_pose_fails(self):
        with self.assertRaises(ValueError):
            ate_se3(self.reference[:-1], self.reference)

    def test_nonfinite_fails(self):
        pred = self.reference.copy()
        pred[1, 0, 3] = np.nan
        with self.assertRaises(ValueError):
            ate_se3(pred, self.reference)

    def test_duplicate_and_invalid_ids_fail(self):
        for ids in ([0, 0, 1], [0, 1, 9], [0., 1., 2.], [0, 1]):
            with self.assertRaises(ValueError):
                ate_se3(self.reference, self.reference, ids)

    def test_upstream_registration_argument_binding(self):
        source = Path(__file__).resolve().parents[2] / 'src/gsr/solver.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'viewpoint_localizer')
        names = [a.arg for a in function.args.args]
        defaults = [ast.literal_eval(x) for x in function.args.defaults]
        required = len(names) - len(defaults)
        signature = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if i < required else defaults[i-required]) for i, n in enumerate(names)])
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'viewpoint_localizer']
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual(len(call.args), 5)
            self.assertFalse(call.keywords)
            self.assertEqual(call.args[3].id, 'tau_registration')
            self.assertEqual(call.args[4].id, 'w_limit')
        bound = signature.bind('camera', 'map', 1e-3, 50., 3.)
        bound.apply_defaults()
        self.assertEqual(bound.arguments['use_variance'], 50.)
        self.assertEqual(bound.arguments['tau_registration'], 3.)
        self.assertEqual(bound.arguments['w_limit'], 3.)


if __name__ == '__main__':
    unittest.main()
