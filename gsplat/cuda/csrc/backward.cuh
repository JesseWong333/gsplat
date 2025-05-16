#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. vT J -> R(n)
__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float* __restrict__ conics,
    const float3* __restrict__ v_xyz,
    const float* __restrict__ v_depth,
    const float* __restrict__ v_conic,
    float* __restrict__ v_cov3d,
    float3* __restrict__ v_mean3d,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat
);


__global__ void nd_rasterize_backward_sum_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const float3* __restrict__ pts,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const int2* __restrict__ tile_bins_pts,
    
    const float* __restrict__ prob_outputs,
    const float* __restrict__ sum_outputs,
    
    const float3* __restrict__ xys,
    const float* __restrict__ conics,
    // const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const float* __restrict__ background,
    const float* __restrict__ v_output,
    float3* __restrict__ v_xyz,
    float* __restrict__ v_conic,
    // float* __restrict__ v_rgb,
    float* __restrict__ v_opacity
    // float* __restrict__ workspace
);


__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *v_cov3d,
    float3 &v_scale,
    float4 &v_quat
);
