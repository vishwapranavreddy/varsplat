"""Opt-in synchronous diagnostic timings. Nested rows are inclusive, not additive."""
from contextlib import contextmanager
from functools import wraps
import time


class Profiler:
    def __init__(self, synchronize, enabled=True):
        self.synchronize = synchronize
        self.enabled = enabled
        self.rows = []
        self.depth = 0

    @contextmanager
    def span(self, stage, **metadata):
        if not self.enabled:
            yield
            return
        self.synchronize()
        start = time.perf_counter()
        depth = self.depth
        self.depth += 1
        status = 'completed'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            self.synchronize()
            seconds = time.perf_counter() - start
            self.depth -= 1
            self.rows.append(dict(stage=stage, seconds=seconds, nesting_depth=depth,
                                  inclusive=True, status=status, **metadata))

    def wrap(self, obj, name, stage, frame=False):
        original = getattr(obj, name)

        @wraps(original)
        def measured(*args, **kwargs):
            metadata = {'frame_id': int(args[0])} if frame and args else {}
            with self.span(stage, **metadata):
                return original(*args, **kwargs)

        setattr(obj, name, measured)


def instrument_stages(slam, profiler):
    import open3d as o3d
    profiler.wrap(o3d.pipelines.registration, 'global_optimization', 'loop.pose_graph_optimization')
    profiler.wrap(slam.tracker, 'track', 'tracking', frame=True)
    profiler.wrap(slam.mapper, 'map', 'mapping', frame=True)
    for name in ('compute_seeding_mask', 'seed_new_gaussians', 'grow_submap', 'optimize_submap'):
        profiler.wrap(slam.mapper, name, 'mapping.' + name)
    for name in ('update_submaps_info', 'detect_closure', 'pairwise_registration', 'construct_pose_graph', 'loop_closure', 'analyse_pgo'):
        profiler.wrap(slam.loop_closer, name, 'loop.' + name)
    for name in ('save_current_submap', 'apply_correction_to_submaps', 'update_keyframe_poses', 'start_new_submap'):
        profiler.wrap(slam, name, 'submaps.' + name)
