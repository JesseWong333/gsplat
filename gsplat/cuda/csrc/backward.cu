#include "backward.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
namespace cg = cooperative_groups;

__global__ void nd_rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float* __restrict__ workspace // 新版本无workspace
) {
    if (channels > MAX_REGISTER_CHANNELS && workspace == nullptr) {
        return;
    }
    // current naive implementation where tile data loading is redundant
    // TODO tile data should be shared between tile threads
    int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    if (i >= img_size.y || j >= img_size.x) {
        return;
    }

    // which gaussians get gradients for this pixel
    int2 range = tile_bins[tile_id];
    // df/d_out for this pixel
    const float *v_out = &(v_output[channels * pix_id]);
    const float v_out_alpha = v_output_alpha[pix_id];
    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[MAX_REGISTER_CHANNELS] = {0.f};
    float *S;
    if (channels <= MAX_REGISTER_CHANNELS) {
        S = &buffer[0];
    } else {
        S = &workspace[channels * pix_id];
    }
    int bin_final = final_index[pix_id];

    // iterate backward to compute the jacobians wrt rgb, opacity, mean2d, and
    // conic recursively compute T_{n-1} from T_n, where T_i = prod(j < i) (1 -
    // alpha_j), and S_{n-1} from S_n, where S_j = sum_{i > j}(rgb_i * alpha_i *
    // T_i) df/dalpha_i = rgb_i * T_i - S_{i+1| / (1 - alpha_i)
    for (int idx = bin_final - 1; idx >= range.x; --idx) {
        const int32_t g = gaussians_ids_sorted[idx];
        const float3 conic = conics[g];
        const float2 center = xys[g];
        const float2 delta = {center.x - px, center.y - py};
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        const float opac = opacities[g];
        const float vis = __expf(-sigma);
        const float alpha = min(0.99f, opac * vis);
        if (alpha < 1.f / 255.f) {
            continue;
        }

        // compute the current T for this gaussian
        const float ra = 1.f / (1.f - alpha);
        T *= ra;
        // rgb = rgbs[g];
        // update v_rgb for this gaussian
        const float fac = alpha * T;
        float v_alpha = 0.f;
        for (int c = 0; c < channels; ++c) {
            // gradient wrt rgb
            atomicAdd(&(v_rgb[channels * g + c]), fac * v_out[c]);
            // contribution from this pixel
            v_alpha += (rgbs[channels * g + c] * T - S[c] * ra) * v_out[c];
            // contribution from background pixel
            v_alpha += -T_final * ra * background[c] * v_out[c];
            // update the running sum
            S[c] += rgbs[channels * g + c] * fac;
        }
        v_alpha += T_final * ra * v_out_alpha;
        // update v_opacity for this gaussian
        atomicAdd(&(v_opacity[g]), vis * v_alpha);

        // compute vjps for conics and means
        // d_sigma / d_delta = conic * delta
        // d_sigma / d_conic = delta * delta.T
        const float v_sigma = -opac * vis * v_alpha;

        atomicAdd(&(v_conic[g].x), 0.5f * v_sigma * delta.x * delta.x);
        atomicAdd(&(v_conic[g].y), 0.5f * v_sigma * delta.x * delta.y);
        atomicAdd(&(v_conic[g].z), 0.5f * v_sigma * delta.y * delta.y);
        atomicAdd(
            &(v_xy[g].x), v_sigma * (conic.x * delta.x + conic.y * delta.y)
        );
        atomicAdd(
            &(v_xy[g].y), v_sigma * (conic.y * delta.x + conic.z * delta.y)
        );
    }
}

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

__global__ void rasterize_backward_kernel(
    const unsigned tile_bounds_x,
    const unsigned tile_bounds_y,
    const unsigned img_size_x,
    const unsigned img_size_y,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds_x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size_x + j, img_size_x * img_size_y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size_y && j < img_size_x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float3 rgbs_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    const float v_out_alpha = v_output_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());  // 这个线程束中最大的一个 index of last gaussian to contribute to this pixel
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - BLOCK_SIZE * b;
        int batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            // if (idx > 10000){
            //     printf("idx %d \n", idx);
            // }
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            // printf("g_id %d \n", g_id);
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float3 conic;
            float vis;
            if(valid){
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.99f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // compute the current T for this gaussian
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += (rgb.x * T - buffer.x * ra) * v_out.x;
                v_alpha += (rgb.y * T - buffer.y * ra) * v_out.y;
                v_alpha += (rgb.z * T - buffer.z * ra) * v_out.z;

                v_alpha += T_final * ra * v_out_alpha;
                // contribution from background pixel
                v_alpha += -T_final * ra * background.x * v_out.x;
                v_alpha += -T_final * ra * background.y * v_out.y;
                v_alpha += -T_final * ra * background.z * v_out.z;
                // update the running sum
                buffer.x += rgb.x * fac;
                buffer.y += rgb.y * fac;
                buffer.z += rgb.z * fac;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                        0.5f * v_sigma * delta.x * delta.y, 
                                        0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_opacity_local = vis * v_alpha;
            }
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
                
                atomicAdd(v_opacity + g, v_opacity_local);
            }
        }
    }
}

__global__ void rasterize_video_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const float time,
    const float vis_thresold,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ means_t,
    const float* __restrict__ lambda,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float* __restrict__ v_means_t,
    float* __restrict__ v_lambda
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float2 time_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float3 rgbs_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    const float v_out_alpha = v_output_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - BLOCK_SIZE * b;
        int batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            time_batch[tr] = {lambda[g_id], means_t[g_id]};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float3 delta;
            float3 conic;
            float vis;
            float lambda;
            if(valid){
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                float2 time_params = time_batch[t];
                opac = xy_opac.z;
                lambda = time_params.x;
                delta = {xy_opac.x - px, xy_opac.y - py, time - time_params.y};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                float decay = 0.5 * lambda * delta.z * delta.z;
                vis = __expf(-sigma-decay);
                alpha = min(0.99f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f || decay > vis_thresold) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_lambda_local = 0.f;
            float v_means_t_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // compute the current T for this gaussian
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += (rgb.x * T - buffer.x * ra) * v_out.x;
                v_alpha += (rgb.y * T - buffer.y * ra) * v_out.y;
                v_alpha += (rgb.z * T - buffer.z * ra) * v_out.z;

                v_alpha += T_final * ra * v_out_alpha;
                // contribution from background pixel
                v_alpha += -T_final * ra * background.x * v_out.x;
                v_alpha += -T_final * ra * background.y * v_out.y;
                v_alpha += -T_final * ra * background.z * v_out.z;
                // update the running sum
                buffer.x += rgb.x * fac;
                buffer.y += rgb.y * fac;
                buffer.z += rgb.z * fac;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                        0.5f * v_sigma * delta.x * delta.y, 
                                        0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_opacity_local = vis * v_alpha;
                v_lambda_local = v_sigma * 0.5 * delta.z * delta.z;
                v_means_t_local = -v_sigma * delta.z * lambda;
            }
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            warpSum(v_means_t_local, warp);
            warpSum(v_lambda_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
                
                atomicAdd(v_opacity + g, v_opacity_local);
                atomicAdd(v_means_t + g, v_means_t_local);
                atomicAdd(v_lambda + g, v_lambda_local);
            }
        }
    }
}


__global__ void rasterize_backward_sum_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    // float T_final = final_Ts[pix_id];
    // float T = T_final;
    // the contribution from gaussians behind the current one
    // float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float3 rgbs_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    // const float v_out_alpha = v_output_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>()); // find max bin_final, 再之外的高斯点可以不处理了
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - BLOCK_SIZE * b;
        int batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float3 conic;
            float vis;
            if(valid){
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(1.f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // compute the current T for this gaussian
                // float ra = 1.f / (1.f - alpha);
                // T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += rgb.x * v_out.x;
                v_alpha += rgb.y * v_out.y;
                v_alpha += rgb.z * v_out.z;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                        0.5f * v_sigma * delta.x * delta.y,   // 这一项也有 0.5？
                                        0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_opacity_local = vis * v_alpha;
            }
            warpSum3(v_rgb_local, warp);  // 一个线程束先做一次 reduce， 再写；而不是每个线程都写回梯度，可以减少主内存压力
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
                atomicAdd(v_opacity + g, v_opacity_local);
            }
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
    const float3* __restrict__ xys,
    const float* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ background,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xyz,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
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

     // df/d_out for this pixel
     const float *v_out = &(v_output[CHANNELS * pix_id]);

    __shared__ int32_t id_batch[N_THREADS];
    __shared__ float4 xyz_opacity_batch[N_THREADS];
    __shared__ float conic_batch[N_THREADS*6];
    // 将全部的数据放入高斯点信息放入共享内存中，容量不够；颜色直接从全局内存中读；to-do:一个替代做法是只渲染有限个，再最后加一个线性层
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
            xy_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            PRAGMA_UNROLL
            for (int i = 0; i < 6; ++i) {
                conic_batch[tr*6 + i] = conics[g_id*6 + i];
            }
            // for(int c = 0; c < CHANNELS; ++c)
            //     rgbs_batch[tr*CHANNELS + c] = rgbs[g_id*CHANNELS + c];
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
            float3 point_pts = {0.0f,0.0f,0.0f};
            float v_out = 0.0f;
            if (render_pixel_inside) {
                point_pts = pts[pts_idx];
                v_out = v_output[pts_idx];
            }
        
            backend_render_one_pixel_of_one_batch_gaussian(
                point_pts,
                v_out,
                id_batch,
                conic_batch,
                xyz_opacity_batch,
                colors,
                num_gaussians_curr_batch,
                render_pixel_inside,
                v_xyz,
                v_conic,
                v_rgb,
                v_opacity,
            );

        }
        
    }
}

__device__ void backward_one_pixel_of_one_batch_gaussian(
    const float3 point, // 输出像素的位置
    const float* __restrict__ v_out, // 输出的像素的梯度
    const int32_t* id_batch, 
    const float* conic_batch, 
    const float4* xyz_opacity_batch, 
    const float* __restrict__ colors,
    const int num_gaussians,
    const int render_pixel_inside, // 当前的这个点渲染是否有效
    // out
    float3* __restrict__ v_xyz,
    float* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_opacity
){

    // 计算一个像素的梯度对one_batch_gaussian, 要写整个batch_size个高斯点的梯度，如何优化, 
    // 使用线程束优化， 先在线程之间 reduce， 避免了每个线程都对主内存进行写

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    for (int t = 0; t < num_gaussians; ++t) {
        int valid = render_pixel_inside;
        float* conic = &(conic_batch[6 * t]);
        float3 xyz_opac = xyz_opacity_batch[t];
        float opac = xyz_opac.w;

        const float3 delta = {xyz_opac.x - point.x, xyz_opac.y - point.y, xyz_opac.z - point.z};

        // calculate sigma in 3D
        const float sigma = 0.5f * (conic[0] * delta.x * delta.x +
                                    conic[3] * delta.y * delta.y +
                                    conic[5] * delta.z * delta.z) +
                            conic[1] * delta.x * delta.y +
                            conic[2] * delta.x * delta.z +
                            conic[4] * delta.y * delta.z;

        float vis = __expf(-sigma);
        float alpha = opac * vis;

        if (sigma < 0.f) {
            valid = 0;
        }
        
        // todo: 即使这里加了一个判断，线程间一定会出现不同步，这里的 warp 判断作用感觉不大
        // bingo: 作用是 warp.thread_rank() == 0这个写入线程一定会走到最后
        // if all threads are inactive in this warp, skip this loop； 
        if(!warp.any(valid)){
            continue;
        }

        float v_rgb_local[CHANNELS] = {0.f};
        float v_conic_local[6] = {0.f};
        float3 v_xyz_local = {0.f, 0.f, 0.f};
        float v_opacity_local = 0.f;
        if(valid){
            // 对一个高斯点的rgb颜色的导数
            const float fac = vis;
            PRAGMA_UNROLL
            for (int c = 0; c < CHANNELS; ++c) {
                v_rgb_local[c] = fac * v_out[c];
            }
            
            // 对alpha，alpha = opac * vis // alpha中间变量
            float v_alpha = 0.f;
            int32_t g = id_batch[t];
            const float *c_ptr = rgbs + g * CHANNELS; 
            PRAGMA_UNROLL
            for (int c = 0; c < CHANNELS; ++c){
                v_alpha += c_ptr[c] * v_out[c];
            }
            
            // 对 sigma 协方差矩阵的逆的导数
            const float v_sigma = -opac * vis * v_alpha;

            // 参照前面的calculate sigma in 3D求逆; 是否每一项都要 0.5f? 对称矩阵
            v_conic_local[0] = 0.5f * v_sigma * delta.x * delta.x;
            v_conic_local[1] = 0.5f * v_sigma * delta.x * delta.y;
            v_conic_local[2] = 0.5f * v_sigma * delta.x * delta.z;
            v_conic_local[3] = 0.5f * v_sigma * delta.y * delta.y;
            v_conic_local[4] = 0.5f * v_sigma * delta.y * delta.z;
            v_conic_local[5] = 0.5f * v_sigma * delta.z * delta.z;

            // 同样参照前面的calculate sigma in 3D求逆;
            v_xyz_local = {
                v_sigma * (conic[0] * delta.x + conic[1] * delta.y + conic[2] * delta.z),
                v_sigma * (conic[1] * delta.x + conic[3] * delta.y + conic[4] * delta.z),
                v_sigma * (conic[2] * delta.x + conic[4] * delta.y + conic[5] * delta.z)
            };

            v_opacity_local = vis * v_alpha;
        }
        // 线程束间 reduce
        warpSum<CHANNELS, float>(v_rgb_local, warp);
        warpSum<6, float>(v_conic_local, warp);
        warpSum3(v_xyz_local, warp);
        warpSum(v_opacity_local, warp);
        // 使用一个线程写回主内存
        if (warp.thread_rank() == 0) {
            int32_t g = id_batch[t];

            float *v_rgb_ptr = (float *)(v_rgb) + CHANNELS * g;
            PRAGMA_UNROLL
            for (int c = 0; c < CHANNELS; ++c) {
                atomicAdd(v_rgb_ptr + c, v_rgb_local[c]);
            }

            float *v_conic_ptr = (float *)(v_conic) + 6 * g;
            PRAGMA_UNROLL
            for (int i = 0; i < 6; ++i){
                atomicAdd(v_conic_ptr + i, v_conic_local[i]);
            }
          
            float *v_xyz_ptr = (float *)(v_xyz) + 3 * g;
            atomicAdd(v_xy_ptr, v_xyz_local.x);
            atomicAdd(v_xy_ptr + 1, v_xyz_local.y);
            atomicAdd(v_xy_ptr + 2, v_xyz_local.z);

            atomicAdd(v_opacity + g, v_opacity_local);
        }
    }
}


__global__ void rasterize_backward_sum_general_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ betas,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float* __restrict__ v_beta
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    // float T_final = final_Ts[pix_id];
    // float T = T_final;
    // the contribution from gaussians behind the current one
    // float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float3 rgbs_batch[BLOCK_SIZE];
    __shared__ float beta_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    // const float v_out_alpha = v_output_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - BLOCK_SIZE * b;
        int batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
            beta_batch[tr] = betas[g_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float3 conic;
            float vis;
            if(valid){
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float conicPart = conic.x * delta.x * delta.x + conic.z * delta.y * delta.y + 2.f * conic.y * delta.x * delta.y; 
                float sigma = 0.5f * pow(conicPart, beta_batch[t]/2.f);
                vis = __expf(-sigma);
                alpha = min(1.f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_beta_local = 0.f;
            const float conicPart = conic.x * delta.x * delta.x + conic.z * delta.y * delta.y + 2.f * conic.y * delta.x * delta.y;
            const float sigma = 0.5f * pow(conicPart, beta_batch[t]/2.f);
            const float v_beta_inter = 0.5f * pow(conicPart, beta_batch[t]/2.f-1);
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // compute the current T for this gaussian
                // float ra = 1.f / (1.f - alpha);
                // T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += rgb.x * v_out.x;
                v_alpha += rgb.y * v_out.y;
                v_alpha += rgb.z * v_out.z;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * v_beta_inter * delta.x * delta.x, 
                                        0.5f * v_sigma * v_beta_inter * delta.x * delta.y, 
                                        0.5f * v_sigma * v_beta_inter * delta.y * delta.y};
                v_xy_local = {v_sigma * v_beta_inter * (conic.x * delta.x + conic.y * delta.y), 
                                    v_sigma * v_beta_inter * (conic.y * delta.x + conic.z * delta.y)};
                v_beta_local = v_sigma * 0.5f * sigma * log(conicPart); 
                v_opacity_local = vis * v_alpha;
            }
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            warpSum(v_beta_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
                atomicAdd(v_opacity + g, v_opacity_local);
                atomicAdd(v_beta + g, v_beta_local);
            }
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
    float *cur_conics = &(conics[6 * idx]);
    float *cur_v_conic = &(v_conic[6 * idx]);
    float cur_v_cov3d[6];
    cov3d_to_conic_vjp(cur_conics[idx], cur_v_conic[idx], cur_v_cov3d[idx]);
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

// output space: 2D covariance, input space: cov3d
__device__ void project_cov3d_ewa_vjp(
    const float3& __restrict__ mean3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float3& __restrict__ v_cov2d,
    float3& __restrict__ v_mean3d,
    float* __restrict__ v_cov3d
) {
    // viewmat is row major, glm is column major
    // upper 3x3 submatrix
    // clang-format off
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    // clang-format on
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;
    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    // clang-format off
    glm::mat3 J = glm::mat3(
        fx * rz,         0.f,             0.f,
        0.f,             fy * rz,         0.f,
        -fx * t.x * rz2, -fy * t.y * rz2, 0.f
    );
    glm::mat3 V = glm::mat3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    // cov = T * V * Tt; G = df/dcov = v_cov
    // -> d/dV = Tt * G * T
    // -> df/dT = G * T * Vt + Gt * T * V
    glm::mat3 v_cov = glm::mat3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );
    // clang-format on

    glm::mat3 T = J * W;
    glm::mat3 Tt = glm::transpose(T);
    glm::mat3 Vt = glm::transpose(V);
    glm::mat3 v_V = Tt * v_cov * T;
    glm::mat3 v_T = v_cov * T * Vt + glm::transpose(v_cov) * T * V;

    // vjp of cov3d parameters
    // v_cov3d_i = v_V : dV/d_cov3d_i
    // where : is frobenius inner product
    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    // compute df/d_mean3d
    // T = J * W
    glm::mat3 v_J = v_T * glm::transpose(W);
    float rz3 = rz2 * rz;
    glm::vec3 v_t = glm::vec3(
        -fx * rz2 * v_J[2][0],
        -fy * rz2 * v_J[2][1],
        -fx * rz2 * v_J[0][0] + 2.f * fx * t.x * rz3 * v_J[2][0] -
            fy * rz2 * v_J[1][1] + 2.f * fy * t.y * rz3 * v_J[2][1]
    );
    // printf("v_t %.2f %.2f %.2f\n", v_t[0], v_t[1], v_t[2]);
    // printf("W %.2f %.2f %.2f\n", W[0][0], W[0][1], W[0][2]);
    v_mean3d.x += (float)glm::dot(v_t, W[0]);
    v_mean3d.y += (float)glm::dot(v_t, W[1]);
    v_mean3d.z += (float)glm::dot(v_t, W[2]);
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
