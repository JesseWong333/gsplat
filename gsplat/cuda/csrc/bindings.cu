#include "backward.cuh"
#include "bindings.h"
#include "forward.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

namespace cg = cooperative_groups;


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
    const float cube_z,
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
) {
    float3 img_size_dim3;
    img_size_dim3.x = cube_x;
    img_size_dim3.y = cube_y;
    img_size_dim3.z = cube_z;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);
   
    // Triangular covariance.
    torch::Tensor cov3d_d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor xys_d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));

    project_gaussians_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        img_size_dim3,
        tile_bounds_dim3,
        clip_thresh,
        // Outputs.
        cov3d_d.contiguous().data_ptr<float>(),
        (float3 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        cov3d_d, xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}

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
) {
    float3 img_size_dim3;
    img_size_dim3.x = cube_x;
    img_size_dim3.y = cube_y;
    img_size_dim3.z = cube_z;

    // Triangular covariance.
    torch::Tensor v_cov3d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean3d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_quat =
        torch::zeros({num_points, 4}, means3d.options().dtype(torch::kFloat32));

    project_gaussians_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        img_size_dim3,
        cov3d.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        (float *)conics.contiguous().data_ptr<float>(),
        (float3 *)v_xyz.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float *)v_conic.contiguous().data_ptr<float>(),
        // Outputs.
        v_cov3d.contiguous().data_ptr<float>(),
        (float3 *)v_mean3d.contiguous().data_ptr<float>(),
        (float3 *)v_scale.contiguous().data_ptr<float>(),
        (float4 *)v_quat.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov3d, v_mean3d, v_scale, v_quat);
}

std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        cum_tiles_hit.contiguous().data_ptr<int32_t>(),
        tile_bounds_dim3,
        // Outputs.
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_tile_bin, int num_intersects, const torch::Tensor &isect_ids_sorted
) {
    CHECK_INPUT(isect_ids_sorted);
    torch::Tensor tile_bins = torch::zeros(
        {num_tile_bin, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    return tile_bins;
}

torch::Tensor get_tile_bin_edges_pts_tensor(
    int num_tile_bin, int num_pts, const torch::Tensor &tile_ids_sorted
){
    CHECK_INPUT(tile_ids_sorted);
    torch::Tensor tile_bins = torch::zeros(
        {num_tile_bin, 2}, tile_ids_sorted.options().dtype(torch::kInt32)
    );
    get_tile_bin_edges_pts<<<
        (num_pts + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_pts,
        tile_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    return tile_bins;
}



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
    // const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(pts);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(tile_bins_pts)
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    // CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    // dim3 block_dim3;
    // block_dim3.x = std::get<0>(block);
    // block_dim3.y = std::get<1>(block);
    // block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    // const int channels = colors.size(1);

    // 现在要渲染的大小是由 Pts 指定
    const int rendering_size = pts.size(0);

    // 默认 logistic 值为
    torch::Tensor out_img = torch::zeros(
        {rendering_size}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {rendering_size}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {rendering_size}, xys.options().dtype(torch::kInt32)
    );
   
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    nd_rasterize_forward_sum<<<tile_bounds_dim3, N_THREADS, 0, stream>>>(
        tile_bounds_dim3,
        img_size_dim3,
        (float3 *) pts.contiguous().data_ptr<float>(),
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (int2 *)tile_bins_pts.contiguous().data_ptr<int>(),
        (float3 *)xys.contiguous().data_ptr<float>(),
        conics.contiguous().data_ptr<float>(),
        // colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        out_img.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, final_Ts, final_idx);
}

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        // torch::Tensor, // dL_dcolors
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
        const torch::Tensor &out, // 渲染的输出
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        // const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &v_output // dL_dout_color
    ) {

    CHECK_INPUT(pts);
    CHECK_INPUT(gaussians_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(tile_bins_pts)
    CHECK_INPUT(out);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    // CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    // dim3 block_dim3;
    // block_dim3.x = std::get<0>(block);
    // block_dim3.y = std::get<1>(block);
    // block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int num_points = xys.size(0);  // 高斯点数
    // const int channels = colors.size(1);

    torch::Tensor v_xyz = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 6}, xys.options());
    // torch::Tensor v_colors =
    //     torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    nd_rasterize_backward_sum_kernel<<<tile_bounds_dim3, N_THREADS, 0, stream>>>(
        tile_bounds_dim3,
        img_size_dim3,
        (float3 *) pts.contiguous().data_ptr<float>(),
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (int2 *)tile_bins_pts.contiguous().data_ptr<int>(),
        out.contiguous().data_ptr<float>(),
        (float3 *)xys.contiguous().data_ptr<float>(),
        conics.contiguous().data_ptr<float>(),
        // colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>(),
        v_output.contiguous().data_ptr<float>(),
        // out
        (float3 *)v_xyz.contiguous().data_ptr<float>(),
        v_conic.contiguous().data_ptr<float>(),
        // v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_xyz, v_conic, v_opacity);
}