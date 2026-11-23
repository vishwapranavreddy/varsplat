"""Strict SE(3), no-scale trajectory evaluation; independent of CUDA imports."""
import numpy as np


def ate_se3(estimated, reference, frame_ids=None):
    estimated, reference = np.asarray(estimated), np.asarray(reference)
    if estimated.shape != reference.shape or estimated.ndim != 3 or estimated.shape[1:] != (4, 4):
        raise ValueError("Require matching N x 4 x 4 trajectories; no silent truncation")
    if not np.isfinite(estimated).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite trajectory; report failure rather than dropping poses")
    for poses in (estimated, reference):
        if not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-5):
            raise ValueError("Invalid homogeneous pose")
        rotations = poses[:, :3, :3]
        if not np.allclose(rotations.swapaxes(1, 2) @ rotations, np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
            raise ValueError("Invalid SO(3) rotation")
    ids = np.arange(len(estimated)) if frame_ids is None else np.asarray(frame_ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) < 3 or len(np.unique(ids)) != len(ids):
        raise ValueError("Need at least three unique integer frame IDs")
    if ids.min() < 0 or ids.max() >= len(estimated):
        raise ValueError("Frame IDs outside trajectory")
    x, y = estimated[ids, :3, 3].astype(np.float64), reference[ids, :3, 3].astype(np.float64)
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = y.mean(0) - rotation @ x.mean(0)
    errors = np.linalg.norm(x @ rotation.T + translation - y, axis=1)
    return {"rmse_m": float(np.sqrt(np.mean(errors**2))), "compared_pose_pairs": len(ids),
            "alignment": "SE3", "scale": 1.0, "frame_ids": ids.tolist(),
            "alignment_rotation": rotation.tolist(), "alignment_translation": translation.tolist()}
