"""Python bindings for custom Cuda functions"""

from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects, bin_pts


def rasterize_gaussians_sum(
    pts: Float[Tensor, "*batch 3"],
    xys: Float[Tensor, "*batch 3"],
    depths: Float[Tensor, "*batch 1"],  # not used
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 6"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    # semantics: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    cube_x: int,
    cube_y: int,
    cube_z: int,
    BLOCK_X: int,
    BLOCK_Y: int,
    BLOCK_Z: int, 
    lidar_mins: Tuple[float, float, float],
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: Optional[bool] = False,
) -> Tensor:
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xyzs (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 3D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        cube_x (int): length in x axis of the rendered cube.
        cube_y (int): length in y axis of the rendered cube.
        cube_z (int): length in z axis of the rendered cube.
        background (Tensor): background color
        return_alpha (bool): whether to return alpha channel

    Returns:
        A Tensor:

        - **out_img** (Tensor): N-dimensional rendered output image.
        - **out_alpha** (Optional[Tensor]): Alpha channel of the rendered output image.
    """

    # if colors.dtype == torch.uint8:
    #     # make sure colors are float [0,1]
    #     colors = colors.float() / 255

     
    if background is None:
        background = torch.ones( 10 ).to(xys.device)

    if xys.ndimension() != 2 or xys.size(1) != 3:
        raise ValueError("xys must have dimensions (N, 3)")

    # if semantics.ndimension() != 2:
    #     raise ValueError("semantics must have dimensions (N, D)")

    return _RasterizeGaussiansSum.apply(
        pts.contiguous(),
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        # semantics.contiguous(),
        opacity.contiguous(),
        cube_x,
        cube_y,
        cube_z,
        BLOCK_X, 
        BLOCK_Y,
        BLOCK_Z,
        lidar_mins,
        background.contiguous(),
        return_alpha,
    )


class _RasterizeGaussiansSum(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        pts: Float[Tensor, "*batch 3"], # points to rendering
        xys: Float[Tensor, "*batch 3"],
        depths: Float[Tensor, "*batch 1"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 6"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        # colors: Float[Tensor, "*batch channels"],
        opacity: Float[Tensor, "*batch 1"],
        cube_x: int,
        cube_y: int,
        cube_z: int,
        BLOCK_X: int,
        BLOCK_Y: int, 
        BLOCK_Z: int,
        lidar_mins: Tuple[float, float, float],
        background: Optional[Float[Tensor, "channels"]] = None,
        return_alpha: Optional[bool] = False,
    ) -> Tensor:
        num_points = xys.size(0)

        tile_bounds = (
            (cube_x + BLOCK_X - 1) // BLOCK_X,
            (cube_y + BLOCK_Y - 1) // BLOCK_Y,
            (cube_z + BLOCK_Z - 1) // BLOCK_Z,
        )
        block = (BLOCK_X, BLOCK_Y, BLOCK_Z)
        img_size = (cube_x, cube_y, cube_z)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

    
        if num_intersects < 1:
            rendering_out = (
                torch.zeros(pts.shape[0], device=xys.device) # 
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
            tile_bins = torch.zeros(0, 2, device=xys.device)
            final_Ts = torch.zeros(cube_x, cube_y, cube_z, device=xys.device)
            final_idx = torch.zeros(cube_x, cube_y, cube_z, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
            )
            
            pts = pts - torch.tensor(lidar_mins).to(pts.device)
            pts_sorted, sorted_indices, inv_sorted_indices, tile_bins_pts = bin_pts(pts, tile_bounds, block)
      
            rendering_out_sorted, prod_outs_sorted, sum_outs_sorted = _C.nd_rasterize_sum_forward(
                pts_sorted,
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                tile_bins_pts,
                xys,
                conics,
                # colors,
                opacity,
                background,
            )
            
            rendering_out = rendering_out_sorted[inv_sorted_indices]

        ctx.cube_x = cube_x
        ctx.cube_y = cube_y
        ctx.cube_z = cube_z
        
        ctx.BLOCK_X = BLOCK_X
        ctx.BLOCK_Y = BLOCK_Y
        ctx.BLOCK_Z = BLOCK_Z

        ctx.tile_bounds = tile_bounds

        ctx.num_intersects = num_intersects
        ctx.save_for_backward(
            prod_outs_sorted,
            sum_outs_sorted,
            pts_sorted,
            gaussian_ids_sorted,
            tile_bins,
            tile_bins_pts,
            xys,
            conics,
            # colors,
            opacity,
            background,
            sorted_indices
        )

        return rendering_out

    @staticmethod
    def backward(ctx, v_out_img):
        
        cube_x = ctx.cube_x
        cube_y = ctx.cube_y
        cube_z = ctx.cube_z
        
        BLOCK_X = ctx.BLOCK_X
        BLOCK_Y = ctx.BLOCK_Y
        BLOCK_Z = ctx.BLOCK_Z

        tile_bounds = ctx.tile_bounds
        
        num_intersects = ctx.num_intersects

        (
            prod_outs_sorted,
            sum_outs_sorted,
            pts_sorted,
            gaussian_ids_sorted,
            tile_bins,
            tile_bins_pts,
            xys,
            conics,
            # colors,
            opacity,
            background,
            sorted_indices
        ) = ctx.saved_tensors
        
        # v_out_img 是无序的梯度
        v_out_img = v_out_img[sorted_indices] # 有序

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            # v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)

        else:
            v_xy, v_conic, v_opacity = _C.nd_rasterize_sum_backward(
                pts_sorted,
                tile_bounds,
                (BLOCK_X, BLOCK_Y, BLOCK_Z),
                (cube_x, cube_y, cube_z),
                gaussian_ids_sorted,
                tile_bins,
                tile_bins_pts,
                prod_outs_sorted,
                sum_outs_sorted,
                xys,
                conics,
                # colors,
                opacity,
                background,
                v_out_img,
            )

        return (
            None,
            v_xy,  # xys
            None,  # depths
            None,  # radii
            v_conic,  # conics
            None,  # num_tiles_hit
            # v_colors,  # colors
            v_opacity,  # opacity
            None,  # cube_x
            None,  # cube_y
            None,  # cube_z
            None,  # block_x
            None,  # block_y
            None,  # block_z
            None,  # lidar_mins 
            None,  # background
            None,  # return_alpha
        )
