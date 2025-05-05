#include "cuda_runtime.h"
#include "forward.cuh"
#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii

compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &A);  // 这个函数可以不暴露


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    const float cube_x,
    const float cube_y,
    const float cube_z, // 现在渲染的是以 pixel 的整数吗, 不是的, 是以 m 为单位的小数等
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    const float cube_x,
    const float cube_y,
    const float cube_z,
    torch::Tensor &cov3d,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xyz,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic
);


std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
);

torch::Tensor get_tile_bin_edges_tensor(
    int num_tile_bin,
    int num_intersects,
    const torch::Tensor &isect_ids_sorted
);

torch::Tensor get_tile_bin_edges_pts_tensor(
    int num_tile_bin,
    int num_pts,
    const torch::Tensor &tile_ids_sorted
);



std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> nd_rasterize_forward_sum_tensor(
    const torch::Tensor &pts,
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &tile_bins_pts,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
);

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    nd_rasterize_backward_sum_tensor(
        const torch::Tensor &pts,
        const std::tuple<int, int, int> tile_bounds,
        const std::tuple<int, int, int> block,
        const std::tuple<int, int, int> img_size,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &tile_bins_pts,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &v_output, // dL_dout_color
    );
    