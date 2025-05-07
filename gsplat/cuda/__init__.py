from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda




nd_rasterize_sum_forward = _make_lazy_cuda_func("nd_rasterize_sum_forward")
nd_rasterize_sum_backward = _make_lazy_cuda_func("nd_rasterize_sum_backward")

project_gaussians_forward = _make_lazy_cuda_func("project_gaussians_forward")
project_gaussians_backward = _make_lazy_cuda_func("project_gaussians_backward")

map_gaussian_to_intersects = _make_lazy_cuda_func("map_gaussian_to_intersects")
get_tile_bin_edges = _make_lazy_cuda_func("get_tile_bin_edges")
get_tile_bin_edges_pts = _make_lazy_cuda_func("get_tile_bin_edges_pts")
