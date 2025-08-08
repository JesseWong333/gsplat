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
import time

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
        self.block_coords = torch.from_numpy(kwargs.get("block_coords", None)).cuda().float() # N, 3
        self.num_blocks = self.block_coords.shape[0]
        
        self.H, self.W, self.L = kwargs["H"], kwargs["W"], kwargs["L"]  # Note, here H, W, L are the dimension of x, y, z
        self.BLOCK_W, self.BLOCK_H, self.BLOCK_L = kwargs["BLOCK_W"], kwargs["BLOCK_H"], kwargs["BLOCK_L"]
        # todo
        assert self.BLOCK_W == self.BLOCK_H == self.BLOCK_L, "BLOCK_W, BLOCK_H, BLOCK_L should be equal"
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            (self.L + self.BLOCK_L - 1) // self.BLOCK_L,
        )
        
        self.lidar_mins = kwargs.get("lidar_mins", [0., 0., 0.])
        
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.num_blocks, self.num_points_per_block, 3) - 0.5)))  # 通过与后面的thanhh函数，得到-1到1之间的数. 做了范围限制
        # self._xyz = nn.Parameter(torch.atanh(0.95 + 0.01 * (torch.rand(self.init_num_points, 3) - 0.5))) # 不好的坐标初始化
        
        self._scaling = nn.Parameter(torch.log(self.BLOCK_W / 3 * torch.rand(self.num_blocks * self.num_points_per_block, 3)))

        self.register_buffer('_opacity', torch.ones((self.num_blocks * self.num_points_per_block, 1)))
        # self._opacity = nn.Parameter(torch.logit(0.5 * torch.ones(self.init_num_points, 1))) # 限制在0-1之间
        # self._opacity = nn.Parameter(10 * torch.rand(self.init_num_points, 1))  # 结合 exp 保证 > 0
        # self._opacity = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._rotation = nn.Parameter(random_quat_tensor(self.num_blocks * self.num_points_per_block))

        self._features_dc = nn.Parameter(torch.rand(self.num_blocks * self.num_points_per_block, kwargs.get("num_channel", 7)))
        # self.register_buffer('_features_dc', torch.ones((self.init_num_points, 1)))

    @property
    def get_xyz(self):
        relative_xyx = 0.5 * torch.tanh(self._xyz) + 0.5  # (-1, 1)  -> (0, 1) (num_blocks, num_points_per_block, 3)
        global_xyx  = (self.block_coords[:, None, :] + relative_xyx) * self.BLOCK_H  # 坐标转换到全局坐标系
        global_xyx = global_xyx / torch.tensor([self.H, self.W, self.L], device=global_xyx.device) # 归一化到 (0, 1)
        global_xyx = 2* global_xyx - 1  # 转换到 (-1, 1)
        return global_xyx.view(self.num_blocks * self.num_points_per_block, 3)  # (num_blocks * num_points_per_block, 3)

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
                                                                                       self.get_rotation, self.H, self.W, self.L,
                                                                                            self.tile_bounds)
        print("num_tiles_hit_ave: {}".format(num_tiles_hit.float().mean().item()))
        print("num_tiles_hit_max: {}".format(num_tiles_hit.float().max().item()))
        print("radii_min: {}".format(radii.float().min().item()))
        print("radii_ave: {}".format(radii.float().mean().item()))
        print("radii_max: {}".format(radii.float().max().item()))
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


if __name__ == '__main__':
    
    # bounds=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    bounds = [1000, 1000, 1000]
    voxel_size = 1
    block_resolution = 16
    
    mins = [0., 0., 0.]
    
    block_size = voxel_size * block_resolution
    
    
    # to-do： 参数调节 1) 内部有太多的点填满了，只表面的 surface 的是不是更好  2）改为 res=16
    sparse_file = "/data/datasets/synthetic_room_dataset_with_meshes/rooms_08/00000335_voxel_1000_res_16.npz"
    # sparse_file = "./samples/1a04_1000_16.npz"
    
    hashmap = o3c.HashMap.load(sparse_file)
    
    block_coords = hashmap.key_tensor().numpy() # N, 3 
    block_values = hashmap.value_tensor(0).numpy() # N, 16, 16, 16
    block_semantics = hashmap.value_tensor(1).numpy() # N, 16, 16, 16
    
    block_semantics += 2 
    
    # 0 empty_classes, 1 wall; 2 '04256520', 3 '03636649', 4 '03001627', 5 '04379243', 6 '02933112'
    
    num_channel = 7  # 占据或者不占据
    Gaussian_points_per_block = 8 # 高斯点
    
    # 使用 sparse voxel gaussian
    gaussian_model = GaussianSSC(Gaussian_points_per_block=Gaussian_points_per_block, 
                                 block_coords = block_coords,
                                 H = bounds[0], W = bounds[1], L = bounds[2], 
                                 BLOCK_W = block_resolution, BLOCK_H = block_resolution, BLOCK_L = block_resolution, 
                                 lidar_mins=mins).cuda()
    
    steps = 2000

    loss_fn = nn.CrossEntropyLoss().cuda()
    # loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([1.])).cuda()
    # loss_fn = nn.BCELoss(weight=torch.tensor([1.])).cuda()

    optimizer = torch.optim.Adam(gaussian_model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    gaussian_model.train()

    # 向量化计算所有块的最小坐标
    block_mins = block_coords * block_size  # Shape: (B, 3)
    
    # 生成相对坐标（在块内的坐标）
    coords = np.linspace(0, block_size, block_resolution, endpoint=False)
    x_rel, y_rel, z_rel = np.meshgrid(coords, coords, coords, indexing='ij')
    relative_points = np.stack([x_rel, y_rel, z_rel], axis=-1).reshape(-1, 3)  # (R³, 3) where R=block_resolution
    
    # 为每个块添加相对坐标
    batch_sample_points = block_mins[:, np.newaxis, :] + relative_points[np.newaxis, :, :]  # (B, R³, 3)
    batch_sample_points = torch.from_numpy(batch_sample_points.reshape(-1, 3)).cuda().float()  # (B*R³, 3)
    targetes = torch.from_numpy(block_semantics).reshape(-1).cuda().long()  #    N, 16, 16, 16 -> N * 16 * 16 * 16
        
    progress_bar = tqdm(range(steps), desc="Training")
    for i in progress_bar:
       
        # random sampling
        start_time = time.time()
        indices = torch.randint(0, batch_sample_points.shape[0], (1000000,))
        sample_points = batch_sample_points[indices]
        sampled_target = targetes[indices]
        
        sampling_time = time.time()
        
        out = gaussian_model.forward(sample_points)
        
        forward_time = time.time()
        
        loss = loss_fn(out, sampled_target)
        # torch.nn.utils.clip_grad_norm_(gaussian_model.parameters(), max_norm=1.0)
        loss.backward()   
        
        backward_time = time.time()
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_description(f"Step {i}, Loss: {loss.item():.6f}, Sampling time: {sampling_time - start_time:.6f}, Forward time: {forward_time - sampling_time:.6f}, Backward time: {backward_time - forward_time:.6f}")

# samping and save

start_time = time.time()
outputs = gaussian_model.forward(batch_sample_points)# N * 16 * 16 * 16, num_classes

rendering_result = outputs.argmax(dim=-1)

print("Forward time:", time.time() - start_time)

rendering_result = rendering_result.reshape(-1, block_resolution, block_resolution, block_resolution).cpu().numpy()  # N, 16, 16, 16

rendering_result = rendering_result.astype(np.int8)  # 转换为 int8 类型

hashmap = o3c.HashMap(50000,
                    key_dtype=o3c.int64,
                    key_element_shape=(3),
                    value_dtype=(o3c.int8),  # 多个元素加s 
                    value_element_shape=(block_resolution, block_resolution, block_resolution),  # 每个格子4096， 对应cuda线程256*16
                    device=o3c.Device("cpu:0"))
    
hashmap.insert(block_coords, rendering_result)  # 插入数据
hashmap.save("./samples/sparse_voxel_gaussian_rendering.npz")  # 保存到文件
