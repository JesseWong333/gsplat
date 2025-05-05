"""Python bindings for 3D gaussian projection"""

from typing import Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


def project_gaussians(
    means3d: Float[Tensor, "*batch 3"],
    scales: Float[Tensor, "*batch 3"],
    glob_scale: float,
    quats: Float[Tensor, "*batch 4"],
    cube_x: int,
    cube_y: int,
    cube_z: int,
    tile_bounds: Tuple[int, int, int],
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, Tensor]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in quaternion [w,x,y,z] format.
       cube_x (int): length in x axis of the rendered cube.
       cube_y (int): length in y axis of the rendered cube.
       cube_z (int): length in z axis of the rendered cube.
       tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, tiles.z).
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
        - **cov3d** (Tensor): 3D covariances.
    """
    return _ProjectGaussians.apply(
        means3d.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        cube_x,
        cube_y,
        cube_z,
        tile_bounds,
        clip_thresh,
    )


class _ProjectGaussians(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means3d: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale: float,
        quats: Float[Tensor, "*batch 4"],
        cube_x: int,
        cube_y: int,
        cube_z: int,
        tile_bounds: Tuple[int, int, int],
        clip_thresh: float = 0.01,
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        (
            cov3d,
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_forward(
            num_points,
            means3d,
            scales,
            glob_scale,
            quats,
            cube_x,
            cube_y,
            cube_z,
            tile_bounds,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.cube_x = cube_x
        ctx.cube_y = cube_y
        ctx.cube_z = cube_z
        ctx.num_points = num_points
        ctx.glob_scale = glob_scale

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            cov3d,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit, cov3d)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit, v_cov3d):
        (
            means3d,
            scales,
            quats,
            cov3d,
            radii,
            conics,
        ) = ctx.saved_tensors

        (v_cov3d, v_mean3d, v_scale, v_quat) = _C.project_gaussians_backward(
            ctx.num_points,
            means3d,
            scales,
            ctx.glob_scale,
            quats,
            ctx.cube_x,
            ctx.cube_y,
            ctx.cube_z,
            cov3d,
            radii,
            conics,
            v_xys,
            v_depths,
            v_conics,
        )

        # Return a gradient for each input. 需要跟 forward的输入 一一对应
        return (
            # means3d: Float[Tensor, "*batch 3"],
            v_mean3d,
            # scales: Float[Tensor, "*batch 3"],
            v_scale,
            # glob_scale: float,
            None,
            # quats: Float[Tensor, "*batch 4"],
            v_quat,
            None,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # img_length: int,
            None,
            # tile_bounds: Tuple[int, int, int],
            None,
            # clip_thresh,
            None,
        )
