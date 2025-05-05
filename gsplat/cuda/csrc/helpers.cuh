#include "config.h"
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <iostream>

#define PRAGMA_UNROLL _Pragma("unroll")

inline __device__ float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5f;
}

inline __device__ void get_bbox(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    uint2 &bb_min,
    uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_tile_bbox(
    const float2 pix_center,
    const float pix_radius,
    const dim3 tile_bounds,
    uint2 &tile_min,
    uint2 &tile_max
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius / (float)BLOCK_X, pix_radius / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

inline __device__ void get_tile_bbox_3d(
    const float3 point_center,
    const float point_radius,
    const dim3 tile_bounds,
    uint3 &tile_min,
    uint3 &tile_max
){
    float3 tile_center = {
        point_center.x / (float)BLOCK_X,
        point_center.y / (float)BLOCK_Y,
        point_center.z / (float)BLOCK_Z
    };

    float3 tile_radius = {
        point_radius / (float)BLOCK_X,
        point_radius / (float)BLOCK_Y,
        point_radius / (float)BLOCK_Z
    };

    // Calculate the bounding box in tile space
    tile_min.x = min(max(0, (int)(tile_center.x - tile_radius.x)), tile_bounds.x);
    tile_max.x = min(max(0, (int)(tile_center.x + tile_radius.x + 1)), tile_bounds.x);

    tile_min.y = min(max(0, (int)(tile_center.y - tile_radius.y)), tile_bounds.y);
    tile_max.y = min(max(0, (int)(tile_center.y + tile_radius.y + 1)), tile_bounds.y);

    tile_min.z = min(max(0, (int)(tile_center.z - tile_radius.z)), tile_bounds.z);
    tile_max.z = min(max(0, (int)(tile_center.z + tile_radius.z + 1)), tile_bounds.z);

}

inline __device__ bool
compute_cov2d_bounds(const float3 cov2d, float3 &conic, float &radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

inline __device__ bool
compute_cov3d_bounds(const float cov3d[6], float conic[6]) {
    // cov3d[0] = x (m11), cov3d[1] = y (m12), cov3d[2] = z (m13)
    // cov3d[3] = w (m22), cov3d[4] = u (m23), cov3d[5] = v (m33)

    // 计算 3x3 矩阵的行列式
    float det = cov3d[0] * (cov3d[3] * cov3d[5] - cov3d[4] * cov3d[4]) -
                cov3d[1] * (cov3d[1] * cov3d[5] - cov3d[2] * cov3d[4]) +
                cov3d[2] * (cov3d[1] * cov3d[4] - cov3d[2] * cov3d[3]);

    if (det == 0.f) {
        return false;  // 矩阵不可逆
    }

    float inv_det = 1.f / det;
    // 计算逆矩阵（对称矩阵，只存储上三角部分）
    conic[0] = (cov3d[3] * cov3d[5] - cov3d[4] * cov3d[4]) * inv_det;  // m11
    conic[1] = (cov3d[2] * cov3d[4] - cov3d[1] * cov3d[5]) * inv_det;  // m12
    conic[2] = (cov3d[1] * cov3d[4] - cov3d[2] * cov3d[3]) * inv_det;  // m13
    conic[3] = (cov3d[0] * cov3d[5] - cov3d[2] * cov3d[2]) * inv_det;  // m22
    conic[4] = (cov3d[2] * cov3d[1] - cov3d[0] * cov3d[4]) * inv_det;  // m23
    conic[5] = (cov3d[0] * cov3d[3] - cov3d[1] * cov3d[1]) * inv_det;  // m33
    
    return true;
}

// compute vjp from df/d_conic to df/c_cov2d
inline __device__ void cov2d_to_conic_vjp(
    const float3 &conic, const float3 &v_conic, float3 &v_cov2d
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic  # ？？？
    glm::mat2 X = glm::mat2(conic.x, conic.y, conic.y, conic.z);
    glm::mat2 G = glm::mat2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    glm::mat2 v_Sigma = -X * G * X;
    v_cov2d.x = v_Sigma[0][0];
    v_cov2d.y = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d.z = v_Sigma[1][1];
}

inline __device__ void cov3d_to_conic_vjp(
    const float conic[6], const float v_conic[6], float v_cov3d[6]
) {
    // 构造对称矩阵X和G
    glm::mat3 X(
        conic[0], conic[1], conic[2],
        conic[1], conic[3], conic[4],
        conic[2], conic[4], conic[5]
    );
    glm::mat3 G(
        v_conic[0], v_conic[1], v_conic[2],
        v_conic[1], v_conic[3], v_conic[4],
        v_conic[2], v_conic[4], v_conic[5]
    );
    
    // 计算梯度矩阵
    glm::mat3 temp = X * G;
    glm::mat3 v_Sigma = -temp * X;
    
    // 提取上三角元素并处理对称性
    v_cov3d[0] = v_Sigma[0][0];
    v_cov3d[1] = v_Sigma[0][1] + v_Sigma[1][0];
    v_cov3d[2] = v_Sigma[0][2] + v_Sigma[2][0];
    v_cov3d[3] = v_Sigma[1][1];
    v_cov3d[4] = v_Sigma[1][2] + v_Sigma[2][1];
    v_cov3d[5] = v_Sigma[2][2];
}

// helper for applying R * p + T, expect mat to be ROW MAJOR
inline __device__ float3 transform_4x3(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline __device__ float4 transform_4x4(const float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline __device__ float2 project_pix(
    const float *mat, const float3 p, const dim3 img_size, const float2 pp
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, img_size.x, pp.x), ndc2pix(p_proj.y, img_size.y, pp.y)
    };
}

// given v_xy_pix, get v_xyz
inline __device__ float3 project_pix_vjp(
    const float *mat, const float3 p, const dim3 img_size, const float2 v_xy
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);

    float3 v_ndc = {0.5f * img_size.x * v_xy.x, 0.5f * img_size.y * v_xy.y};
    float4 v_proj = {
        v_ndc.x * rw, v_ndc.y * rw, 0., -(v_ndc.x + v_ndc.y) * rw * rw
    };
    // df / d_world = df / d_cam * d_cam / d_world
    // = v_proj * P[:3, :3]
    return {
        mat[0] * v_proj.x + mat[4] * v_proj.y + mat[8] * v_proj.z,
        mat[1] * v_proj.x + mat[5] * v_proj.y + mat[9] * v_proj.z,
        mat[2] * v_proj.x + mat[6] * v_proj.y + mat[10] * v_proj.z
    };
}

inline __device__ glm::mat3 quat_to_rotmat(const float4 quat) {
    // quat to rotation matrix
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    // glm matrices are column-major
    return glm::mat3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

// inline __device__ glm::mat3 rotor_to_rotmat(const float4 rot) {
//     // quat to rotation matrix
//     float s = rsqrtf(
//         rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w
//     );
//     float x = rot.x * s;
//     float y = rot.y * s;
//     float z = rot.z * s;
//     float w = rot.w * s;

//     // glm matrices are column-major
//     return glm::mat3(
//         x * x - y * y - z * z + w * w,
//         -2.f * (x * y + w * z),
//         2.f * (y * w - x * z),
//         2.f * (x * y - w * z),
//         x * x - y * y + z * z - w * w,
//         -2.f * (y * z + w * x),
//         2.f * (y * w + x * z),
//         2.f * (x * w - y * z),
//         x * x + y * y - z * z - w * w
//     );
// }



inline __device__ float4
quat_to_rotmat_vjp(const float4 quat, const glm::mat3 v_R) {
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x =
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              );
    // x element in y field
    v_quat.y =
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        );
    // y element in z field
    v_quat.z =
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        );
    // z element in w field
    v_quat.w =
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        );
    return v_quat;
}

inline __device__ glm::mat3
scale_to_mat(const float3 scale, const float glob_scale) {
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// inline __device__ glm::mat3
// inverse_scale_to_mat(const float3 scale, const float glob_scale) {
//     glm::mat3 S = glm::mat3(1.f);
//     S[0][0] = 1 / (glob_scale * scale.x);
//     S[1][1] = 1 / (glob_scale * scale.y);
//     S[2][2] = 1 / (glob_scale * scale.z);
//     return S;
// }

inline __device__ glm::mat3
triangular_mat(const float3 diag_elements, const float3 non_diag_elements) {
    glm::mat3 L = glm::mat3(1.f);
    L[0][0] = diag_elements.x;
    L[1][1] = diag_elements.y;
    L[2][2] = diag_elements.z;
    L[1][0] = non_diag_elements.x;
    L[2][0] = non_diag_elements.y;
    L[2][1] = non_diag_elements.z;
    return L;
}


inline __device__ glm::mat2
scale_to_mat2d(const float2 scale) {
    glm::mat2 S = glm::mat2(1.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    return S;
}

inline __device__ glm::mat2 rotmat2d(const float rot) {
    // quat to rotation matrix
    float cosr = cos(rot);
    float sinr = sin(rot);

    glm::mat2 R = glm::mat2(cosr);
    R[0][1] = -sinr;
    R[1][0] = sinr;

    // glm matrices are column-major
    return R;
}

inline __device__ glm::mat2 rotmat2d_gradient(const float rot) {
    // quat to rotation matrix
    float cosr = cos(rot);
    float sinr = sin(rot);

    glm::mat2 R = glm::mat2(-sinr);
    R[0][1] = -cosr;
    R[1][0] = cosr;

    // glm matrices are column-major
    return R;
}

// device helper for culling near points
inline __device__ bool clip_near_plane(
    const float3 p, const float *viewmat, float3 &p_view, float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}
