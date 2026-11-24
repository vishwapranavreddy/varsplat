"""Exercise original refinement/render metrics on a tiny synthetic fixture, not benchmark data."""
from itertools import cycle


def check():
    import numpy as np
    import open3d as o3d
    import torch
    from torch.utils.data import DataLoader
    from pytorch_msssim import ms_ssim
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from src.evaluation.evaluate_merged_map import RenderFrames, refine_global_map
    from src.utils.utils import get_render_settings, render_gaussian_model

    class Fixture:
        intrinsics = np.array([[100., 0., 96.], [0., 100., 96.], [0., 0., 1.]])
        def __len__(self): return 2
        def __getitem__(self, i):
            return i, np.full((192, 192, 3), 128, np.uint8), np.full((192, 192), 2., np.float32), np.eye(4)

    dataset = Fixture()
    frames = RenderFrames(dataset, np.repeat(np.eye(4)[None], 2, axis=0), 192, 192, 100., 100.)
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 32), np.linspace(-1.5, 1.5, 32))
    points = np.stack([x.ravel(), y.ravel(), np.full(x.size, 2.)], axis=-1)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.full_like(points, .5))
    model = refine_global_map(cloud, cycle(DataLoader(frames, batch_size=1, shuffle=False)), 2)
    render = render_gaussian_model(model, get_render_settings(192, 192, dataset.intrinsics, np.eye(4)))['color'].detach().clamp(0, 1)[None]
    target = torch.full_like(render, 128/255.)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()(render, target)
    ssim = ms_ssim(render, target, data_range=1., size_average=True)
    if not torch.isfinite(lpips) or not torch.isfinite(ssim): raise ValueError('Non-finite metric smoke test')
    torch.cuda.synchronize()
    return {'status': 'PASSED', 'fixture': 'synthetic; not a quality benchmark',
            'refinement_iterations': 2, 'checks': ['original_refinement_with_collated_settings', 'alex_lpips_offline_weights', 'ms_ssim']}
