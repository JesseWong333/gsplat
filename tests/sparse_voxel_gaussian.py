# 训练的时候使用稀疏的 voxel 采样

# semantic 版本测试

import torch
import torch.nn as nn
import numpy as np
import trimesh
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize_sum import rasterize_gaussians_sum
from tqdm import tqdm
import math
import open3d as o3d
import open3d.core as o3c
from spconv.pytorch.utils import PointToVoxel

np.random.seed(1)
torch.manual_seed(1)

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N, 1)
    v = torch.rand(N, 1)
    w = torch.rand(N, 1)
    return torch.cat(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

class GaussianSSC(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_points_per_block = kwargs.get("Gaussian_points_per_block", 4)
        self.block_coords = kwargs.get("block_coords", None)  # N, 3
        self.num_blocks = self.block_coords.shape[0]
        self.tile_bounds = kwargs.get("block_d", [125, 125, 125])  # [dx, dy, dz]
        self.block_resolution = kwargs.get("block_resolution", 8)
        self.voxel_size = kwargs.get("voxel_size", 0.001)
        self.block_size = np.array(self.voxel_size * self.block_resolution)
        
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.num_blocks, self.num_points_per_block, 3) - 0.5)))  # 通过与后面的thanhh函数，得到-1到1之间的数. 做了范围限制
        # self._xyz = nn.Parameter(torch.atanh(0.95 + 0.01 * (torch.rand(self.init_num_points, 3) - 0.5))) # 不好的坐标初始化
        
        self._scaling = nn.Parameter(torch.log(torch.rand(self.num_blocks, self.num_points_per_block, 3)))

        self.register_buffer('_opacity', torch.ones((self.num_blocks, self.num_points_per_block, 1)))
        # self._opacity = nn.Parameter(torch.logit(0.5 * torch.ones(self.init_num_points, 1))) # 限制在0-1之间
        # self._opacity = nn.Parameter(10 * torch.rand(self.init_num_points, 1))  # 结合 exp 保证 > 0
        # self._opacity = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._rotation = nn.Parameter(random_quat_tensor(self.num_blocks*self.num_points_per_block))

        self._features_dc = nn.Parameter(torch.rand(self.num_blocks * self.num_points_per_block, kwargs.get("num_channel", 7)))
        # self.register_buffer('_features_dc', torch.ones((self.init_num_points, 1)))

    @property
    def get_xyz(self):
        relative_xyx = 0.5 * torch.tanh(self._xyz) + 0.5  # (-1, 1)  -> (0, 1)
        global_xyx  = 2 * (self.block_coords + relative_xyx) * self.block_size - 1  # 坐标转换到全局坐标系 (0, 1) -> (-1, 1)
        return global_xyx

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)
    
    @property
    def get_opacity(self):
        # return torch.exp(self._opacity) # 保证 > 0
        return self._opacity
    
    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)  # 旋转四元数模长为1
    
    def forward(self, x):
        # rendering: x: N, 3 要渲染block
        
        xys, depths, radii, conics, num_tiles_hit = project_gaussians(self.get_xyz, self.get_scaling, 1, 
                                                                                       self.get_rotation, 1, 1, 1,
                                                                                            self.tile_bounds)
        return rasterize_gaussians_sum(x, xys, depths, radii, conics, num_tiles_hit, 
                                        self._features_dc, 
                                       self.get_opacity, 
                                       self.H, self.W, self.L,
                                       self.BLOCK_W, self.BLOCK_H, self.BLOCK_L,
                                       self.lidar_mins 
                                       )

def load_file(path):
    points_dict = np.load(path)
    points = points_dict['points']
    semantics = points_dict['semantics']
    return points, semantics

class PointQuantizer:   
    def __init__(self, voxel_size, ranges, num_point_features=3, max_num_voxels=8000, max_num_points_per_voxel=1, device=torch.device("cpu:0")):

        self.voxel_size = voxel_size
        self.ranges = ranges
        self.pointToVoxel = PointToVoxel(self.voxel_size, self.ranges, num_point_features, max_num_voxels, max_num_points_per_voxel, device)

    def __call__(self, points):
        voxels, indices, num_per_voxel = self.pointToVoxel(points)
        
        return indices


def get_index(positive_indices, indices):
    """
    获取 positive_indices 在 indices 中的索引
    positive_indices: tensor [N, 3] 
    indices: [M, 3]
    """
    positive_expanded = positive_indices.unsqueeze(1)  # [N, 1, 3]
    indices_expanded = indices.unsqueeze(0)            # [1, M, 3]
    
    # Element-wise comparison
    equal = (positive_expanded == indices_expanded)    # [N, M, 3]
    
    # Check if all 3 coordinates match for each pair
    matches = equal.all(dim=2)                         # [N, M]
    
    # Find the index in indices for each positive_index
    # Returns the index in indices for each positive_indices, or -1 if not found
    index_in_indices = torch.where(matches)[1]         # Get the matching indices
    
    # Create result tensor with -1 as default (for non-found items)
    result = torch.full((positive_indices.shape[0],), -1, dtype=torch.long, device=positive_indices.device)
    
    # Get the indices of positive_indices that have matches
    positive_has_match = torch.where(matches)[0]
    
    # Assign the matching indices
    result[positive_has_match] = index_in_indices
    return result

if __name__ == '__main__':
    
    bounds=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    voxel_size = 0.001
    block_resolution = 8
    
    block_size = voxel_size * block_resolution
    
    block_dx = np.ceil((bounds[3] - bounds[0]) / block_size).astype(int)
    block_dy = np.ceil((bounds[4] - bounds[1]) / block_size).astype(int)
    block_dz = np.ceil((bounds[5] - bounds[2]) / block_size).astype(int)
    
    # to-do： 参数调节 1) 内部有太多的点填满了，只表面的 surface 的是不是更好  2）改为 res=16
    sparse_file = "/data/datasets/synthetic_room_dataset_with_meshes/rooms_08/00000985_voxel_1000_res_8.npz"
    
    hashmap = o3c.HashMap.load(sparse_file)
    
    block_coords = hashmap.key_tensor().numpy() # N, 3 
    block_values = hashmap.value_tensor(0).numpy() # N, 16, 16, 16
    block_semantics = hashmap.value_tensor(1).numpy() # N, 16, 16, 16 
    
    block_semantics += 2 
    
    # 0 empty_classes, 1 wall; 2 '04256520', 3 '03636649', 4 '03001627', 5 '04379243', 6 '02933112'
    
    num_channel = 7  # 占据或者不占据
    Gaussian_points_per_block = 4 # 高斯点
    
    # 使用 sparse voxel gaussian
    gaussian_model = GaussianSSC(Gaussian_points_per_block=Gaussian_points_per_block, 
                                 block_coords=block_coords, 
                                 block_d=[block_dx, block_dy, block_dz], 
                                 voxel_size=voxel_size, 
                                 block_resolution=block_resolution,
                                 num_channel=num_channel).cuda()
    
    steps = 2000

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1, 1, 1, 1, 1, 1])).cuda()
    # loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([1.])).cuda()
    # loss_fn = nn.BCELoss(weight=torch.tensor([1.])).cuda()

    optimizer = torch.optim.Adam(gaussian_model.parameters(), lr=0.005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    gaussian_model.train()

    progress_bar = tqdm(range(steps), desc="Training")
    for i in progress_bar:
    
        sampled_points = torch.from_numpy(valid_points).cuda()
        sampled_target = torch.from_numpy(semantics).cuda()

        out = gaussian_model.forward(sampled_points)
        # out = out.squeeze(1)
        loss = loss_fn(out, sampled_target)
        # torch.nn.utils.clip_grad_norm_(gaussian_model.parameters(), max_norm=1.0)
        loss.backward()   
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_description(f"Step {i}, Loss: {loss.item()}")


    # torch.save(gaussian_model.state_dict(), "./gaussian_model.pth.tar")

# 这个重建效果不好可能是因为缩放了

# samping and save
start = -10
end = 10
step = 0.2
x = np.arange(start, end, step)
y = np.arange(start, end, step)
z = np.arange(start, end, step)

# Generate all combinations of coordinates
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

outputs = gaussian_model.forward(torch.from_numpy(points).cuda().float()) # N * num_classes

# -------------------------------------------------------------
save_points = np.concatenate((points, outputs.cpu().detach().numpy()), axis=-1)
np.save("./samples/points_for_march_cubes_color", save_points) 

pass

# -------------------------------------------------------------
# rendering_result = outputs.argmax(dim=-1)

# mask = rendering_result != 0  # 只保留非空的点
# rendering_result = rendering_result[mask]
# points = points[mask.cpu().detach().numpy()]

# save_points = np.concatenate((points, rendering_result.unsqueeze(-1).cpu().detach().numpy()), axis=-1)

# print(save_points.shape)

# np.save("./samples/points_500_color", save_points) 



