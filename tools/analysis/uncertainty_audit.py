"""CPU diagnostics for M1; no estimator mutation or claimed SLAM improvement.

Inputs are N x 3 RGB residuals/appearance variances in normalized RGB units.
Depth variance must not be passed here: it has different physical units.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def validate(variance, valid):
    v = np.asarray(variance, dtype=np.float64)
    mask = np.asarray(valid)
    if v.ndim != 2 or v.shape[1] != 3 or mask.shape != (len(v),) or mask.dtype != np.bool_:
        raise ValueError('Require N x 3 variance and N boolean validity mask')
    if not mask.any():
        raise ValueError('No valid observations')
    if not np.isfinite(v[mask]).all() or (v[mask] <= 0).any():
        raise ValueError('Selected appearance variances must be finite and positive')
    return v[mask]


def centered_weights(variance, valid, tau, limit, eps=1e-6):
    """Selected-pixel RGB weights with torch.median's lower-middle convention."""
    v = validate(variance, valid)
    if (not all(math.isfinite(x) for x in (tau, limit, eps))
            or tau <= 0 or limit < 1 or eps < 0):
        raise ValueError('Invalid weight parameters')
    logv = np.log(v + eps)
    k = (len(logv)-1)//2
    median = np.partition(logv, k, axis=0)[k]
    # Clipping in log-space is mathematically equivalent and avoids exp overflow.
    return np.exp(np.clip(-(logv-median)/tau, -np.log(limit), np.log(limit)))


def calibration_statistics(residual, variance, valid, bins=8):
    v = validate(variance, valid)
    residual = np.asarray(residual, dtype=np.float64)
    if residual.shape != np.asarray(variance).shape or not np.isfinite(residual[valid]).all():
        raise ValueError('Finite RGB residuals must match the variance shape')
    if type(bins) is not int or bins < 1:
        raise ValueError('Positive integer bin count required')
    r2 = np.square(residual[valid])
    standardized2 = r2/v
    nll = .5*(standardized2 + np.log(v) + math.log(2*math.pi))
    if not np.isfinite(nll).all():
        raise ValueError('Non-finite residual likelihood')
    rows = []
    # Equal-count bins per channel. The discrepancy is descriptive, not an ECE
    # guarantee for correlated SLAM residuals or a calibrated pose covariance.
    for c in range(3):
        channel = []
        for indices in np.array_split(np.argsort(v[:, c], kind='stable'), min(bins, len(v))):
            channel.append(dict(count=len(indices), mean_variance=float(v[indices, c].mean()),
                                mean_squared_residual=float(r2[indices, c].mean())))
        rows.append(channel)
    return dict(valid_observations=len(v), rgb_gaussian_nll_by_channel=nll.mean(axis=0).tolist(),
                standardized_squared_residual_by_channel=standardized2.mean(axis=0).tolist(),
                one_sigma_coverage_by_channel=(standardized2 <= 1).mean(axis=0).tolist(),
                two_sigma_coverage_by_channel=(standardized2 <= 4).mean(axis=0).tolist(),
                variance_bins_by_channel=rows)


def scale_sensitivity(variance, valid, tau, limit, scales=(.1, 10.), eps=1e-6):
    original = centered_weights(variance, valid, tau, limit, eps)
    exact = centered_weights(variance, valid, tau, limit, 0.)
    rows = []
    for scale in scales:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Positive finite variance scale required')
        changed = np.asarray(variance, dtype=np.float64)*scale
        rows.append(dict(scale=scale,
                         max_weight_change=float(np.abs(centered_weights(changed, valid, tau, limit, eps)-original).max()),
                         max_weight_change_without_epsilon=float(np.abs(centered_weights(changed, valid, tau, limit, 0.)-exact).max())))
    return dict(tau=tau, limit=limit, eps=eps, scale_tests=rows,
                interpretation='Uniform variance scaling cancels under median log-centering when eps=0; finite epsilon can break invariance.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', required=True, type=Path,
                   help='NPZ: residual [N,3], variance [N,3], valid [N] bool')
    p.add_argument('--tau', required=True, type=float)
    p.add_argument('--limit', required=True, type=float)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    import hashlib
    with np.load(args.samples, allow_pickle=False) as data:
        r, v, mask = data['residual'], data['variance'], data['valid']
        result = dict(calibration=calibration_statistics(r, v, mask),
                      weighting=scale_sensitivity(v, mask, args.tau, args.limit),
                      sample_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                      scope='Appearance residual audit only; no ATE/time or generalization claim')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
