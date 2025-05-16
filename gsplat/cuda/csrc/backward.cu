#include "backward.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
namespace cg = cooperative_groups;


inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}

template <uint32_t DIM, class T, class WarpT>
inline __device__ void warpSum(T *val, WarpT &warp) {
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<T>());
    }
}

__device__ void backward_one_pixel_of_one_batch_gaussian(
    const float prob_out,
    const float sum_out,
    float3 point, // 输出像素的位置
    const float v_out, // 输出的像素的梯度, channel维
    const int32_t* id_batch, 
    const float* conic_batch, 
    const float4* xyz_opacity_batch, 
    // const float* __restrict__ colors,
    const int num_gaussians,
    const int render_pixel_inside, // 当前的这个点渲染是否有效
    // out
    float3* __restrict__ v_xyz,
    float* __restrict__ v_conic,
    // float* __restrict__ v_rgb,
    float* __restrict__ v_opacity
){

    // 计算一个像素的梯度对one_batch_gaussian, 要写整个batch_size个高斯点的梯度，如何优化, 
    // 使用线程束优化， 先在线程之间 reduce， 避免了每个线程都对主内存进行写

    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    float sigmoid_sum = 1.f / (1 + __expf(-sum_out));
    float term_2 = (1 - prob_out) * sigmoid_sum * (1 - sigmoid_sum);  // (1-A)*S*(1-S)这一项对每个高斯都相同；提前一起做

    for (int t = 0; t < num_gaussians; ++t) {
        int valid = render_pixel_inside;
        const float* conic = &(conic_batch[6 * t]);
        const float4 xyz_opac = xyz_opacity_batch[t];
        const float opac = xyz_opac.w;

        const float3 delta = {xyz_opac.x - point.x, xyz_opac.y - point.y, xyz_opac.z - point.z};

        // calculate sigma in 3D
        const float sigma = 0.5f * (conic[0] * delta.x * delta.x +
                                    conic[3] * delta.y * delta.y +
                                    conic[5] * delta.z * delta.z) +
                            conic[1] * delta.x * delta.y +
                            conic[2] * delta.x * delta.z +
                            conic[4] * delta.y * delta.z;  

        float vis = __expf(-sigma);

        // float alpha = opac * vis; // alpha 即是输出

        if (sigma < 0.f) {
            valid = 0;
        }
        
        // bingo: 作用是 warp.thread_rank() == 0这个写入线程一定会走到最后
        // if all threads are inactive in this warp, skip this loop； 
        if(!warp.any(valid)){
            continue;
        }

        float v_conic_local[6] = {0.f};
        float3 v_xyz_local = {0.f, 0.f, 0.f};
        float v_opacity_local = 0.f;
        if(valid){
            // 对 sigma 协方差矩阵的逆的导数
            // const float v_sigma = - (1 - out) / (1- vis + 1e-9) * vis * v_out;
            float term_1 = - prob_out / (1 - vis + 1e-9) * vis * sigmoid_sum;
            float v_sigma = (term_1 + term_2) * v_out;

            // 参照前面的calculate sigma in 3D求逆; 是否每一项都要 0.5f? 对称矩阵
            v_conic_local[0] = v_sigma * delta.x * delta.x;
            v_conic_local[1] = 0.5f * v_sigma * delta.x * delta.y;
            v_conic_local[2] = 0.5f * v_sigma * delta.x * delta.z;
            v_conic_local[3] = v_sigma * delta.y * delta.y;
            v_conic_local[4] = 0.5f * v_sigma * delta.y * delta.z;
            v_conic_local[5] = v_sigma * delta.z * delta.z;

            // 同样参照前面的calculate sigma in 3D求逆;
            v_xyz_local = {
                v_sigma * (conic[0] * delta.x + conic[1] * delta.y + conic[2] * delta.z),
                v_sigma * (conic[1] * delta.x + conic[3] * delta.y + conic[4] * delta.z),
                v_sigma * (conic[2] * delta.x + conic[4] * delta.y + conic[5] * delta.z)
            };
        
            // v_opacity_local = vis * v_out;

            v_opacity_local = term_2 * vis * v_out;

        }
        // 线程束间 reduce
        // warpSum<CHANNELS, float>(v_rgb_local, warp);

        warpSum<6, float>(v_conic_local, warp);
        warpSum3(v_xyz_local, warp);
        warpSum(v_opacity_local, warp);
        // 使用一个线程写回主内存
        if (warp.thread_rank() == 0) {
            int32_t g = id_batch[t];

            float *v_conic_ptr = (float *)(v_conic) + 6 * g;
            PRAGMA_UNROLL
            for (int i = 0; i < 6; ++i){
                atomicAdd(v_conic_ptr + i, v_conic_local[i]);
            }
          
            float *v_xyz_ptr = (float *)(v_xyz) + 3 * g;
            atomicAdd(v_xyz_ptr, v_xyz_local.x);
            atomicAdd(v_xyz_ptr + 1, v_xyz_local.y);
            atomicAdd(v_xyz_ptr + 2, v_xyz_local.z);

            atomicAdd(v_opacity + g, v_opacity_local);
        }
    }
}

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
) {
    
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().z * tile_bounds.x * tile_bounds.y + block.group_index().y * tile_bounds.x + block.group_index().x;
        
    const int2 range = tile_bins[tile_id];
    const int2 pts_range = tile_bins_pts[tile_id];
 
    const int tr = block.thread_rank();

    const int num_batches = (range.y - range.x + N_THREADS - 1) / N_THREADS;
    int num_points_rendering = (pts_range.y - pts_range.x + N_THREADS - 1) / N_THREADS;

    __shared__ int32_t id_batch[N_THREADS];
    __shared__ float4 xyz_opacity_batch[N_THREADS];
    __shared__ float conic_batch[N_THREADS*6];
    // 将全部的数据放入高斯点信息放入共享内存中，容量不够；颜色直接从全局内存中读；不要颜色
    // __shared__ float rgbs_batch[BLOCK_SIZE * CHANNELS];

    // cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    for (int b = 0; b < num_batches; ++b) {
        block.sync();

        int batch_start = range.x + N_THREADS * b; // 
        int idx = batch_start + tr;

        if (idx < range.y) {
            int32_t g_id = gaussians_ids_sorted[idx];
            id_batch[tr] = g_id;  // id_batch, xy_opacity_batch, conic_batch, rgbs_batch
            const float3 xyz = xys[g_id];
            const float opac = opacities[g_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            PRAGMA_UNROLL
            for (int i = 0; i < 6; ++i) {
                conic_batch[tr*6 + i] = conics[g_id*6 + i];
            }
        }

        block.sync();

        int num_gaussians_curr_batch = min(N_THREADS,range.y - batch_start);
        for(int b_p = 0; b_p < num_points_rendering; ++b_p){
            // 先确定当前线程是渲染哪一个像素
            int pts_batch_start = pts_range.x + N_THREADS * b_p;
            int pts_idx = pts_batch_start + tr;

            int render_pixel_inside  = 1;
            if (pts_idx >= pts_range.y) {
                render_pixel_inside = 0;
            }
            float prod_out = 1.0f;
            float sum_out = 0.0f;
            float3 point_pts = {0.0f,0.0f,0.0f};
            float v_out = 0.0f;
            if (render_pixel_inside) {
                prod_out = prob_outputs[pts_idx];
                sum_out = sum_outputs[pts_idx];

                point_pts = pts[pts_idx];
                v_out = v_output[pts_idx];
            }
        
            backward_one_pixel_of_one_batch_gaussian(
                prod_out,
                sum_out,
                point_pts,
                v_out,
                id_batch,
                conic_batch,
                xyz_opacity_batch,
                // colors,
                num_gaussians_curr_batch,
                render_pixel_inside,
                v_xyz,
                v_conic,
                // v_rgb,
                v_opacity
            );

        }
        
    }
}


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
    // output
    float* __restrict__ v_cov3d,
    float3* __restrict__ v_mean3d,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    
    v_mean3d[idx].x = v_xyz[idx].x * (0.5f * img_size.x);
    v_mean3d[idx].y = v_xyz[idx].y * (0.5f * img_size.y);
    v_mean3d[idx].z = v_xyz[idx].z * (0.5f * img_size.z);

    // get v_cov3d and write it to v_cov3d
    const float *cur_conics = &(conics[6 * idx]);
    const float *cur_v_conic = &(v_conic[6 * idx]);
    float cur_v_cov3d[6];
    cov3d_to_conic_vjp(cur_conics, cur_v_conic, cur_v_cov3d);
    for (int i = 0; i < 6; ++i) {
        v_cov3d[6 * idx + i] = cur_v_cov3d[i];
    }

    // get v_scale and v_quat
    scale_rot_to_cov3d_vjp(
        scales[idx],
        glob_scale,
        quats[idx],
        &(v_cov3d[6 * idx]),
        v_scale[idx],
        v_quat[idx]
    );
}


// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float* __restrict__ v_cov3d,
    float3& __restrict__ v_scale,
    float4& __restrict__ v_quat
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    glm::mat3 v_V = glm::mat3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    glm::mat3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    glm::mat3 v_M = 2.f * v_V * M;
    // glm::mat3 v_S = glm::transpose(R) * v_M;
    v_scale.x = (float)glm::dot(R[0], v_M[0]);
    v_scale.y = (float)glm::dot(R[1], v_M[1]);
    v_scale.z = (float)glm::dot(R[2], v_M[2]);

    glm::mat3 v_R = v_M * S;
    v_quat = quat_to_rotmat_vjp(quat, v_R);
}
