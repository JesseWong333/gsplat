"""Python bindings for binning and sorting gaussians"""

from typing import Tuple

from jaxtyping import Float, Int
from torch import Tensor
import torch

import gsplat.cuda as _C


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 3"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
    )
    return (isect_ids, gaussian_ids)


def get_tile_bin_edges(
    num_tile_bin:int, num_intersects: int, isect_ids_sorted: Int[Tensor, "num_intersects 1"]
) -> Int[Tensor, "num_intersects 2"]:
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
    """
    return _C.get_tile_bin_edges(num_tile_bin, num_intersects, isect_ids_sorted.contiguous())


def compute_cov2d_bounds(
    cov2d: Float[Tensor, "batch 3"]
) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {Tensor, Tensor}:

        - **conic** (Tensor): conic parameters for 2D gaussian.
        - **radii** (Tensor): radii of 2D gaussian projections.
    """
    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _C.compute_cov2d_bounds(num_pts, cov2d.contiguous())


def compute_cumulative_intersects(
    num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 3"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y, z locations of 3D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **isect_ids_unsorted** (Tensor): unique IDs for each gaussian in the form (tile | depth id). 
        - **gaussian_ids_unsorted** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds  # isect_ids, gaussian_ids长度都是扩充后的长度， isect_ids存储tile | depth id； gaussian_ids存储原高斯点的索引
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    # tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted)  # 我觉得这里可能是个bug, 
    tile_bins = get_tile_bin_edges(tile_bounds[0] * tile_bounds[1] * tile_bounds[2], num_intersects, isect_ids_sorted) # range of gaussians IDs hit per tile 
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins


def bin_pts(
    pts: Float[Tensor, "batch 3"],
    tile_bounds: Tuple[int, int, int],
    block: Tuple[int, int, int]
    ) -> Tuple[
        Float[Tensor, "batch 3"],
        Float[Tensor, "batch 1"],
        Float[Tensor, "batch 2"],
    ]:
    """Bin points to tile IDs.
    
    # 现在要渲染的就是 pts 大小, 类似于高斯点的做法，但是无需复制
    # 1. 计算每个点归属的 tile_ID pytorch
    # 2. tile_ID list 进行排序， pts相应排序
    # 3. 对tile_ID list调用 kernel, 类似于get_tile_bin_edges, 得到pts的tile_bin
    # pts: N * 3
    """
    x = pts[:, 0] // block[0]  # N 
    y = pts[:, 1] // block[1]
    z = pts[:, 2] // block[2]
    tile_ids = tile_bounds[0] *tile_bounds[1] * z + tile_bounds[0] * y + x # N
    tile_ids = tile_ids.int()
    
    tile_ids_sorted, sorted_indices = torch.sort(tile_ids)
    pts_sorted = pts[sorted_indices]
    
    inv_sorted_indices = torch.zeros_like(sorted_indices)
    inv_sorted_indices.scatter_(0, sorted_indices, torch.arange(sorted_indices.size(0)).to(sorted_indices.device))
    
    tile_bin = _C.get_tile_bin_edges_pts(tile_bounds[0] * tile_bounds[1] * tile_bounds[2], pts.shape[0], tile_ids_sorted.contiguous()) # tile_bounds[0] * tile_bounds[1] * tile_bounds[2], 2
    return pts_sorted, sorted_indices, inv_sorted_indices, tile_bin

def bin_gaussian_pts(     
                xys,
                tile_bounds,
                block
            ):
    """
    # return: gaussian_ids_sorted, tile_bins
    """
    x = xys[:, 0] // block[0]  # N 
    y = xys[:, 1] // block[1]
    z = xys[:, 2] // block[2]
    tile_ids = tile_bounds[0] *tile_bounds[1] * z + tile_bounds[0] * y + x # N, 每个高斯点的tile_id
    tile_ids = tile_ids.int()
    tile_ids_sorted, sorted_indices = torch.sort(tile_ids)
    
    gaussian_ids = torch.arange(xys.shape[0]).to(xys.device)
    gaussian_ids_sorted = gaussian_ids[sorted_indices].int()
    
    tile_bin = _C.get_tile_bin_edges_pts(tile_bounds[0] * tile_bounds[1] * tile_bounds[2], xys.shape[0], tile_ids_sorted.contiguous()) 
    return gaussian_ids_sorted, tile_bin
