#include "forward.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,  // 没用
    float* __restrict__ covs3d,
    float3* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // compute the projected covariance
    float3 scale = scales[idx];
    float4 quat = quats[idx];

    float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    float radius = fmaxf(fmaxf(scale.x, scale.y), scale.z);
    float conic[6];
    bool ok = compute_cov3d_bounds(cur_cov3d, conic);
    if (!ok)
        return; // zero determinant
    PRAGMA_UNROLL
    for (int i = 0; i < 6; ++i) {
        conics[6 * idx + i] = conic[i];
    }
    
    // compute the mean in world space
    float3 center = {0.5f * img_size.x * means3d[idx].x + 0.5f * img_size.x,
                    0.5f * img_size.y * means3d[idx].y + 0.5f * img_size.y,
                    0.5f * img_size.z * means3d[idx].z + 0.5f * img_size.z}; // 这里转换实际的坐标

    uint3 tile_min, tile_max;
    get_tile_bbox_3d(center, radius, tile_bounds, tile_min, tile_max);
    
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y) * (tile_max.z - tile_min.z);
    if (tile_area <= 0) {
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = 0.0f;
    radii[idx] = (int)radius;
    xys[idx] = center;
}

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float3* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    // get the tile bbox for gaussian
    uint3 tile_min, tile_max;
    float3 center = xys[idx];
    get_tile_bbox_3d(center, radii[idx], tile_bounds, tile_min, tile_max);

    // update the intersection info for all tiles this gaussian hits
    //  E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    int32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];  // cum_tiles_hit是累加的 list； cur_idx是当前的高斯点在扩充后的list起始的位置
    int64_t depth_id = (int64_t) * (int32_t *)&(depths[idx]);
    for (int k = tile_min.z; k < tile_max.z; ++k) {
        for (int i = tile_min.y; i < tile_max.y; ++i) {
            for (int j = tile_min.x; j < tile_max.x; ++j) {
                // tile_id is tile ID and depth as int32
                int64_t tile_id = k * tile_bounds.x * tile_bounds.y + i * tile_bounds.x + j;  // tile within rendering cube 高斯点碰到的一个 tile
                isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
                gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
                ++cur_idx; // handles gaussians that hit more than one tile
            }
        }
    }
}

// kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
__global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t* __restrict__ isect_ids_sorted, int2* __restrict__ tile_bins
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            tile_bins[cur_tile_idx].x = 0;
        if (idx == num_intersects - 1)
            tile_bins[cur_tile_idx].y = num_intersects;
    }
    if (idx == 0)
        return;
    int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {   // 只找交界处的， [tile_id_3, tile_id_3], [tile_id_4, tile_id_4, tile_id_4]
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x = idx;
        return;
    }
}

// to-do：这个项目本来就不需要深度，此函数和上面的函数其实一样
__global__ void get_tile_bin_edges_pts(
    const int num_pts, const int32_t* __restrict__ tile_ids_sorted, int2* __restrict__ tile_bins
){
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_pts)
        return;
    
    int32_t cur_tile_idx = tile_ids_sorted[idx];
    if (idx == 0 || idx == num_pts - 1) {
        if (idx == 0)
            tile_bins[cur_tile_idx].x = 0;
        if (idx == num_pts - 1)
            tile_bins[cur_tile_idx].y = num_pts;
    }
    if (idx == 0)
        return;
    int32_t prev_tile_idx = tile_ids_sorted[idx - 1];
    if(prev_tile_idx != cur_tile_idx) { 
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x = idx;
        return;
    }
}

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
__global__ void nd_rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float* __restrict__ out_img,
    const float* __restrict__ background
) {
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

    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    float T = 1.f;

    // iterate over all gaussians and apply rendering EWA equation (e.q. 2 from
    // paper)
    int idx;
    for (idx = range.x; idx < range.y; ++idx) {
        const int32_t g = gaussian_ids_sorted[idx];
        const float3 conic = conics[g];
        const float2 center = xys[g];
        const float2 delta = {center.x - px, center.y - py};

        // Mahalanobis distance (here referred to as sigma) measures how many
        // standard deviations away distance delta is. sigma = -0.5(d.T * conic
        // * d)
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        const float opac = opacities[g];

        const float alpha = min(0.999f, opac * __expf(-sigma));

        // break out conditions
        if (alpha < 1.f / 255.f) {
            continue;
        }
        const float next_T = T * (1.f - alpha);
        if (next_T <= 1e-4f) {
            // we want to render the last gaussian that contributes and note
            // that here idx > range.x so we don't underflow
            idx -= 1;
            break;
        }
        const float vis = alpha * T;
        for (int c = 0; c < channels; ++c) {
            out_img[channels * pix_id + c] += colors[channels * g + c] * vis;
        }
        T = next_T;
    }
    final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
    final_index[pix_id] =
        (idx == range.y)
            ? idx - 1
            : idx; // index of in bin of last gaussian in this pixel
    for (int c = 0; c < channels; ++c) {
        out_img[channels * pix_id + c] += T * background[c];
    }
}



__global__ void rasterize_forward(
    const unsigned tile_bounds_x,
    const unsigned tile_bounds_y,
    const unsigned img_size_x,
    const unsigned img_size_y,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds_x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size_x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size_y && j < img_size_x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    // if (tile_id > 18){
    //     printf("tile_id: %d", tile_id);
    // }
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load

        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float3 c = colors[g];  // 颜色还是从主存读取的啊
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            T = next_T;
            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        out_img[pix_id] = final_color;
    }
}

__global__ void rasterize_video_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const float time,
    const float vis_thresold,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    const float* __restrict__ means_t,
    const float* __restrict__ lambda,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float2 time_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            time_batch[tr] = {lambda[g_id], means_t[g_id]};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 time_params = time_batch[t];
            const float3 delta = {xy_opac.x - px, xy_opac.y - py, time - time_params.y};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
        
            const float decay = 0.5 * time_params.x * delta.z * delta.z;
            const float alpha = min(0.999f, opac * __expf(-sigma-decay));
            if (sigma < 0.f || alpha < 1.f / 255.f || decay > vis_thresold) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            T = next_T;
            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        out_img[pix_id] = final_color;
    }
}




__global__ void rasterize_forward_sum(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(1.f, opac * __expf(-sigma));  // alpha不超过1
            if (sigma < 0.f || alpha < 1.f / 255.f) { // 控制 float的值, 限制在了最小
                continue;
            }

            int32_t g = id_batch[t];
            const float vis = alpha;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis; // vis - > opcity
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            // T = next_T;
            cur_idx = batch_start + t;
        }
        done = true;
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x; //+ T * background.x;  // 删除了背景
        final_color.y = pix_out.y; //+ T * background.y;
        final_color.z = pix_out.z; //+ T * background.z;
        out_img[pix_id] = final_color;
    }
}


__global__ void nd_rasterize_forward_sum(
    const dim3 tile_bounds,
    const dim3 cube_size,
    const float3* __restrict__ pts, // N * 3
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const int2* __restrict__ tile_bins_pts,
    const float3* __restrict__ xys,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,   // todo: not used
    int* __restrict__ final_index,  // todo: not used
    float* __restrict__ out_img,  // N * C
    const float* __restrict__ background  //todo: may be used in the future
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().z * tile_bounds.x * tile_bounds.y + block.group_index().y * tile_bounds.x + block.group_index().x;
    
    // 得到当前要渲染的位置，现在一个点要渲染多个点
    // unsigned i =
    //     block.group_index().y * block.group_dim().y + block.thread_index().y;
    // unsigned j =
    //     block.group_index().x * block.group_dim().x + block.thread_index().x;

    // float px = (float)j;
    // float py = (float)i;


    // int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    // bool inside = (i < img_size.y && j < img_size.x);
    // bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int2 pts_range = tile_bins_pts[tile_id];
    
    int num_batches = (range.y - range.x + N_THREADS - 1) / N_THREADS;  // 当前 tile高斯点的数量 / tile线程数； 一个线程需要从全局内存中取的高斯点数
    int num_points_rendering = (pts_range.y - pts_range.x + N_THREADS - 1) / N_THREADS; // 一个线程需要渲染的点数

    // 4090： 128SM, 128KB shared memory, 1536 threads， 16 blocks
    // 2080：46 SM， 64 KB， 1024 线程
    // 共享内层 256 * 4 + 256*3*4 + 256*6*4 = 10240 字节 （10k） 每个SM的共享内存总量固定, 每个线程使用的共享内存越少，每个SM可驻留的线程块越多
    // 则 4090 大概 可以 12个线程块*256 线程 = 3072， 超过了最大数量，满载
    // 则 2080 大概可 6 个线程块*256 线程 = 1024， 刚好满载
   
    
    // 并行思路：
    // 1) 每个线程取一个数据（一个高斯点）到共享内存; 此时数据没有取完，可能高斯点更多； 做了一个 batch,  batch_size = BLOCK_SIZE
    // 2) 每个线程使用所有取到的高斯点进行渲染
    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    
    __shared__ int32_t id_batch[N_THREADS];  // 高斯点 id
    __shared__ float4 xyz_opacity_batch[N_THREADS];
    __shared__ float conic_batch[N_THREADS*6];

    int tr = block.thread_rank();
    float pix_out[num_points_rendering][CHANNELS] = {0.f};  // 这个数据有多的
    
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        // if (__syncthreads_count(done) >= N_THREADS) {  // 统计当前线程是否都done了，如果都done了，则跳出循环
        //     break;
        // }
        block.sync();
        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + N_THREADS * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float3 xyz = xys[g_id];
            const float opac = opacities[g_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};

            PRAGMA_UNROLL
            for (int i = 0; i < 6; ++i) {
                conic_batch[tr*6 + i] = conics[g_id*6 + i];
            }
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        int num_gaussians_curr_batch = min(N_THREADS, range.y - batch_start);
        for(int b_p = 0; b_p < num_points_rendering; ++b_p){
            // 先确定当前线程是渲染哪一个像素
            int pts_batch_start = pts_range.x + N_THREADS * b_p;
            int pts_idx = pts_batch_start + tr;
            if (pts_idx >= pts_range.y) {
                continue; // 最后一个 batch 是凑不成N_THREADS个，需要判断
            }
            float3 point_pts = pts[pts_idx];  // 当前线程要渲染的位置

            render_one_pixel_of_one_batch_gaussian(
                point_pts,
                id_batch,
                conic_batch,
                xyz_opacity_batch,
                colors,
                num_gaussians_curr_batch,
                pix_out[b_p]
            );
        }  
    }

    // 复制回主存储
    for(int b_p = 0; b_p < num_points_rendering; ++b_p){
        int pts_batch_start = pts_range.x + N_THREADS * b_p;
        int pts_idx = pts_batch_start + tr;
        if (pts_idx >= pts_range.y) {
            continue;
        }
        PRAGMA_UNROLL
        for (int c = 0; c < CHANNELS; ++c) {
            out_img[pts_idx * CHANNELS + c] = pix_out[b_p][c]; // + T * background[c] no bg
        }
    }
    
    
}

__device__ void render_one_pixel_of_one_batch_gaussian(
    const float3 point,  // 需要渲染的位置
    const int32_t* id_batch, 
    const float* conic_batch, 
    const float4* xyz_opacity_batch, 
    const float* __restrict__ colors,
    const int num_gaussians,
    // out
    float* __restrict__ pix_out,
) {
    for (int t = 0; (t < num_gaussians); ++t) {
        // const float3 conic = conic_batch[t];
        const float *conic = &(conic_batch[6 * t]);
        const float4 xyz_opac = xyz_opacity_batch[t];
        const float opac = xy_opac.w;

        const float3 delta = {xyz_opac.x - point.x, xyz_opac.y - point.y, xyz_opac.z - point.z};

        // calculate sigma in 3D
        const float sigma = 0.5f * (conic[0] * delta.x * delta.x +
                                     conic[3] * delta.y * delta.y +
                                     conic[5] * delta.z * delta.z) +
                             conic[1] * delta.x * delta.y +
                             conic[2] * delta.x * delta.z +
                             conic[4] * delta.y * delta.z;
        
        if (sigma < 0.f) {
            continue;
        }
        
        const float vis = opac * __expf(-sigma);
        int32_t g = id_batch[t];

        const float *c_ptr = colors + g * CHANNELS; // 颜色是直接从主存储里取的
        PRAGMA_UNROLL
        for (int c = 0; c < CHANNELS; ++c) {
            pix_out[c] += c_ptr[c] * vis;  // 不同的线程是写不同位置，无需同步
        }
    }
}


// __global__ void nd_rasterize_forward_sum(
//     const dim3 tile_bounds,
//     const dim3 img_size,
//     const unsigned channels,
//     const int32_t* __restrict__ gaussian_ids_sorted,
//     const int2* __restrict__ tile_bins,
//     const float2* __restrict__ xys,
//     const float3* __restrict__ conics,
//     const float* __restrict__ colors,
//     const float* __restrict__ opacities,
//     float* __restrict__ final_Ts,
//     int* __restrict__ final_index,
//     float* __restrict__ out_img,
//     const float* __restrict__ background
// ) {
//     // current naive implementation where tile data loading is redundant
//     // TODO tile data should be shared between tile threads
//     int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
//     unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
//     unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
//     float px = (float)j;
//     float py = (float)i;
//     int32_t pix_id = i * img_size.x + j;

//     // return if out of bounds
//     if (i >= img_size.y || j >= img_size.x) {
//         return;
//     }

//     // which gaussians to look through in this tile
//     int2 range = tile_bins[tile_id];
//     float T = 1.f;

//     // iterate over all gaussians and apply rendering EWA equation (e.q. 2 from
//     // paper)
//     int idx;
//     for (idx = range.x; idx < range.y; ++idx) {
//         const int32_t g = gaussian_ids_sorted[idx];
//         const float3 conic = conics[g];
//         const float2 center = xys[g];
//         const float2 delta = {center.x - px, center.y - py};

//         // Mahalanobis distance (here referred to as sigma) measures how many
//         // standard deviations away distance delta is. sigma = -0.5(d.T * conic
//         // * d)
//         const float sigma =
//             0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
//             conic.y * delta.x * delta.y;
//         if (sigma < 0.f) {
//             continue;
//         }
//         const float opac = opacities[g];

//         const float alpha = min(1.f, opac * __expf(-sigma));

//         // break out conditions
//         if (alpha < 1.f / 255.f) {
//             continue;
//         }
//         // const float next_T = T * (1.f - alpha);
//         // if (next_T <= 1e-4f) {
//         //     // we want to render the last gaussian that contributes and note
//         //     // that here idx > range.x so we don't underflow
//         //     idx -= 1;
//         //     break;
//         // }
//         const float vis = alpha; //* T;
//         for (int c = 0; c < channels; ++c) {
//             out_img[channels * pix_id + c] += colors[channels * g + c] * vis;
//         }
//         //T = next_T;
//     }
//     final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
//     final_index[pix_id] =
//         (idx == range.y)
//             ? idx - 1
//             : idx; // index of in bin of last gaussian in this pixel
//     // for (int c = 0; c < channels; ++c) {
//     //     out_img[channels * pix_id + c] += T * background[c];
//     // }
// }

__global__ void rasterize_forward_sum_general(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    const float* __restrict__ betas,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float beta_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            beta_batch[tr] = betas[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float conicPart = conic.x * delta.x * delta.x + conic.z * delta.y * delta.y + 2.f * conic.y * delta.x * delta.y;
            const float beta = beta_batch[t];
            const float sigma = 0.5f * pow(conicPart, beta/2);
            const float opac = xy_opac.z;
            const float alpha = min(1.f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }
            int32_t g = id_batch[t];
            const float vis = alpha;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            // T = next_T;
            cur_idx = batch_start + t;
        }
        done = true;
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x; //+ T * background.x;
        final_color.y = pix_out.y; //+ T * background.y;
        final_color.z = pix_out.z; //+ T * background.z;
        out_img[pix_id] = final_color;
    }
}

// device helper to approximate projected 2d cov from 3d mean and cov
__device__ float3 project_cov3d_ewa(
    const float3& __restrict__ mean3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy
) {
    // clip the
    // we expect row major matrices as input, glm uses column major
    // upper 3x3 submatrix
    glm::mat3 W = glm::mat3(
        viewmat[0],
        viewmat[4],
        viewmat[8],
        viewmat[1],
        viewmat[5],
        viewmat[9],
        viewmat[2],
        viewmat[6],
        viewmat[10]
    );
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;

    // clip so that the covariance
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    t.x = t.z * std::min(lim_x, std::max(-lim_x, t.x / t.z));
    t.y = t.z * std::min(lim_y, std::max(-lim_y, t.y / t.z));

    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    glm::mat3 J = glm::mat3(
        fx * rz,
        0.f,
        0.f,
        0.f,
        fy * rz,
        0.f,
        -fx * t.x * rz2,
        -fy * t.y * rz2,
        0.f
    );
    glm::mat3 T = J * W;

    glm::mat3 V = glm::mat3(
        cov3d[0],
        cov3d[1],
        cov3d[2],
        cov3d[1],
        cov3d[3],
        cov3d[4],
        cov3d[2],
        cov3d[4],
        cov3d[5]
    );

    glm::mat3 cov = T * V * glm::transpose(T);

    // add a little blur along axes and save upper triangular elements
    return make_float3(float(cov[0][0]) + 0.3f, float(cov[0][1]), float(cov[1][1]) + 0.3f);
}

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
) {
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);

    glm::mat3 M = R * S;
    glm::mat3 tmp = M * glm::transpose(M);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}
