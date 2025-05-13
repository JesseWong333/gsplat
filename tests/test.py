# 这个版本还是训练不出来

import torch
import torch.nn as nn
import numpy as np
import trimesh
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize_sum import rasterize_gaussians_sum
from tqdm import tqdm
import math

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
        self.init_num_points = kwargs.get("num_points", 2000)
        self.H, self.W, self.L = kwargs["H"], kwargs["W"], kwargs["L"]  # Note, here H, W, L are the dimension of x, y, z
        self.BLOCK_W, self.BLOCK_H, self.BLOCK_L = kwargs["BLOCK_W"], kwargs["BLOCK_H"], kwargs["BLOCK_L"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            (self.L + self.BLOCK_L - 1) // self.BLOCK_L,
        )
        
        self.lidar_mins = kwargs.get("lidar_mins", [-10, -10, -10])
        
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 3) - 0.5)))  # 通过与后面的thanhh函数，得到-1到1之间的数. 做了范围限制

        self._scaling = nn.Parameter(torch.log(torch.rand(self.init_num_points, 3)))

        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        # self._opacity = nn.Parameter(torch.logit(0.5 * torch.ones(self.init_num_points, 1))) # 限制在0-1之间
        # self._opacity = nn.Parameter(10 * torch.rand(self.init_num_points, 1))  # 结合 exp 保证 > 0
        # self._opacity = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._rotation = nn.Parameter(random_quat_tensor(self.init_num_points))

        # self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 2)) 
        # self.register_buffer('_features_dc', torch.ones((self.init_num_points, 1)))

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz) 

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
        # rendering: x: B, N, 3 要渲染的位置
        
        xys, depths, radii, conics, num_tiles_hit = project_gaussians(self.get_xyz, self.get_scaling, 1, 
                                                                                       self.get_rotation, self.H, self.W, self.L,
                                                                                            self.tile_bounds)
        return rasterize_gaussians_sum(x, xys, depths, radii, conics, num_tiles_hit, 
                                        # self._features_dc, 
                                       self.get_opacity, 
                                       self.H, self.W, self.L,
                                       self.BLOCK_W, self.BLOCK_H, self.BLOCK_L,
                                       self.lidar_mins 
                                       )


if __name__ == '__main__':
    
    # num_points = 2000
    # num_channel = 28 + 1  # 有一个为空的的类别 
    # tile_ranges = [256*2, 256, 32] # 这里包括前后
    # lidar_mins = [-51.2, -25.6, -2]  # [0, -25.6, -2, 51.2, 25.6, 4.4]
    # grid_size = 0.2  # 
    
    # ------------------------------------------------------------------------------------------------------------------
    # 单独的一个物体拟合实验
    from plyfile import PlyData
    from glob import glob
    
    file_list = glob("/hd_cache/users/junjie/projects/Gaussian_SSC/data/synthetic_room/points_iou/*.npz")

    valid_points_l = []
    occupancies_l = []
    for file_path in file_list:
        points_dict = np.load(file_path)
        points = points_dict['points']
        occupancies = points_dict['occupancies']
        occupancies = np.unpackbits(occupancies)[:points.shape[0]]

        # mask = occupancies != 0
        # valid_points_l.append(points[mask])

        valid_points_l.append(points)
        occupancies_l.append(occupancies)

    valid_points = np.concatenate(valid_points_l, axis=0).astype(np.float32)  # 1Mpoints
    occupancies = np.concatenate(occupancies_l, axis=0).astype(np.int64)
    
    # 已经 scale 到了[-0.55, 0.55]之间
    valid_points = valid_points * 18   # 大概 [-10， 10]  # 实际位置

    # normlize
    lidar_mins = [-10., -10., -10.]
    grid_size = 1
    
    num_channel = 2  # 占据或者不占据
    num_points = 2000 # 高斯点
    
    # 我这样做其实是一个生成式的3D模型，雷达是采样的点
    gaussian_model = GaussianSSC(num_points=num_points, H = 20, W = 20, L = 20, BLOCK_W = 1, BLOCK_H = 1, BLOCK_L = 1, lidar_mins=lidar_mins).cuda()
    
    steps = 5000

    # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.8, 1]).cuda())
    # loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([1.])).cuda()
    loss_fn = nn.BCELoss(weight=torch.tensor([1.])).cuda()

    optimizer = torch.optim.Adam(gaussian_model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    gaussian_model.train()

    progress_bar = tqdm(range(steps), desc="Training")
    for i in progress_bar:
    
        sampled_points = torch.from_numpy(valid_points).cuda()
        sampled_target = torch.from_numpy(occupancies).cuda().float()

        out = gaussian_model.forward(sampled_points) # N * 1; 渲染就是全 0是初始化的问题
        # out = out.squeeze(1)
        loss = loss_fn(out, sampled_target)
        # torch.nn.utils.clip_grad_norm_(gaussian_model.parameters(), max_norm=1.0)
        loss.backward()   
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_description(f"Step {i}, Loss: {loss.item()}")


    # torch.save(gaussian_model.state_dict(), "./gaussian_model.pth.tar")
    

# samping and save
start = -10.0
end = 10.0
step = 0.1
x = np.arange(start, end, step)
y = np.arange(start, end, step)
z = np.arange(start, end, step)

# Generate all combinations of coordinates
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

rendering_result = gaussian_model.forward(torch.from_numpy(points).cuda().float())

# rendering_result = rendering_result.squeeze(1)

# occupancies = rendering_result.argmax(dim=-1)

mask = rendering_result > 0

# mask = occupancies != 0
mask = mask.cpu().numpy()

points = points[mask]

print(points.shape)

np.save("./samples/points_2000_no_color", points) 

