# 测试一个cube看xyz如何移动

import torch
import torch.nn as nn
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize_sum import rasterize_gaussians_sum
from tqdm import tqdm
import math
import pyvista as pv
from PIL import Image
import io

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

        # self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        # self._opacity = nn.Parameter(torch.logit(0.5 * torch.ones(self.init_num_points, 1))) # 限制在0-1之间
        # self._opacity = nn.Parameter(10 * torch.rand(self.init_num_points, 1))  # 结合 exp 保证 > 0
        self._opacity = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._rotation = nn.Parameter(random_quat_tensor(self.init_num_points))

        # self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 2)) 
        # self.register_buffer('_features_dc', torch.ones((self.init_num_points, 1)))

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz) 

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)
        # 返回 torch.exp(self._scaling) 的结果
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

def create_3d_sphere_grid(size=16, radius_ratio=0.4):
    """
    创建一个16x16x16的球体占据网格
    
    Args:
        size: 网格尺寸 (默认16)
        radius_ratio: 球体半径相对于网格大小的比例
    
    Returns:
        torch.Tensor: 形状为(size, size, size)的二值化球体网格
    """
    # 创建坐标网格
    coords = torch.arange(size)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    
    # 计算中心点和半径
    center = size / 2.0
    radius = size * radius_ratio
    
    # 计算每个点到中心的距离
    distances = torch.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    
    # 创建球体（距离小于半径的点标记为1）
    sphere = (distances <= radius).float()
    
    return sphere

def create_3d_tetrahedron_grid_simple(size=16):
    """
    创建一个更简单的16x16x16四面体占据网格（使用不等式方法）
    
    Args:
        size: 网格尺寸 (默认16)
    
    Returns:
        torch.Tensor: 形状为(size, size, size)的二值化四面体网格
    """
    # 创建坐标网格
    coords = torch.arange(size, dtype=torch.float32)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    
    # 组合所有条件创建四面体
    condition1 = x - y + z >= 0           # 通过(0,0,0),(0,1,1),(1,1,0)的平面
    condition2 = x + y - z >= 0           # 通过(0,0,0),(1,0,1),(1,1,0)的平面
    condition3 = -x + y + z >= 0          # 通过(0,0,0),(0,1,1),(1,0,1)的平面
    condition4 = x + y + z <= 1.5 * (size-1)  # 限制四面体大小
    
    # 另一种更清晰的四面体定义方式
    # 使用标准的四面体顶点：(0,0,0), (1,0,0), (0,1,0), (0,0,1)
    # 然后放大到网格尺寸
    
    # 重新定义更清晰的四面体：
    # 我们创建一个顶点在(0,0,0), (size-1,0,0), (0,size-1,0), (0,0,size-1)的四面体
    condition1 = y + z <= size - 1      # 通过(0,0,0),(size-1,0,0),(0,size-1,0)的平面
    condition2 = x + z <= size - 1      # 通过(0,0,0),(size-1,0,0),(0,0,size-1)的平面
    condition3 = x + y <= size - 1      # 通过(0,0,0),(0,size-1,0),(0,0,size-1)的平面
    condition4 = x >= 0                 # x非负
    condition5 = y >= 0                 # y非负
    condition6 = z >= 0                 # z非负
    
    # 四面体是所有条件都满足的区域
    tetrahedron = (condition1 & condition2 & condition3 & condition4 & condition5 & condition6).float()
    
    return tetrahedron

if __name__ == '__main__':
    
    # ------------------------------------------------------------------------------------------------------------------
    # 测试cube拟合
    cube_x = 16
    cube_y = 16
    cube_z = 16
    
    gt_image = create_3d_tetrahedron_grid_simple(size=cube_x)
        
    # normlize
    mins = [0., 0., 0.]
    
    grid_size = 16
    
    num_channel = 2  # 占据或者不占据
    num_points = 10 # 高斯点
    
    # 我这样做其实是一个生成式的3D模型，雷达是采样的点
    gaussian_model = GaussianSSC(num_points=num_points, H = cube_x, W = cube_y, L = cube_z, BLOCK_W = grid_size, BLOCK_H = grid_size, BLOCK_L = grid_size, lidar_mins=mins).cuda()
    
    steps = 1000

    # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.8, 1]).cuda())
    loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([1.])).cuda()
    # loss_fn = nn.BCELoss(weight=torch.tensor([1.])).cuda()

    optimizer = torch.optim.Adam(gaussian_model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    gaussian_model.train()

    # --------------------------------------------------------------
    start = 0
    end = 16
    step = 1
    x = np.arange(start, end, step)
    y = np.arange(start, end, step)
    z = np.arange(start, end, step)

    # Generate all combinations of coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    target = gt_image[points[:, 0], points[:, 1], points[:, 2]]
    # --------------------------------------------------------------

    progress_bar = tqdm(range(steps), desc="Training")
    
    vis_data = []
    for i in progress_bar:
    
        sampled_points = torch.from_numpy(points).cuda().float()
        sampled_target = target.cuda().float()

        out = gaussian_model.forward(sampled_points) # N * 1; 渲染就是全 0是初始化的问题
        # out = out.squeeze(1)
        loss = loss_fn(out, sampled_target)
        # torch.nn.utils.clip_grad_norm_(gaussian_model.parameters(), max_norm=1.0)
        loss.backward()   
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_description(f"Step {i}, Loss: {loss.item()}")
        if i % 10 == 0:
            vis_data.append(
                {
                    "out": out.sigmoid().cpu().detach().numpy(),     
                    "xyz": gaussian_model.get_xyz.cpu().detach().numpy(),
                    "scale": gaussian_model.get_scaling.cpu().detach().numpy(),
                    "quat": gaussian_model.get_rotation.cpu().detach().numpy(),
                    "opacity": gaussian_model.get_opacity.cpu().detach().numpy(),
                }       
                # out.sigmoid().cpu().detach().numpy() 
            )

# torch.save(gaussian_model.state_dict(), "./gaussian_model.pth.tar")

# visulize
# -------------------------------------------------------------------------------------------------------

def add_reference_grid(plotter, bounds=None, n_ticks=5, color="gray", opacity=0.2):
    """添加立方体网格和坐标轴标签"""
    if bounds is None:
        bounds = np.array(plotter.bounds)
    
    # 主立方体框架
    grid = pv.Box(bounds=bounds)
    plotter.add_mesh(
        grid,
        color=color,
        opacity=opacity,
        style="wireframe",
        line_width=1.5,
        lighting=False,
        label="Reference Grid"
    )
    
    # 手动添加坐标轴标签（替代旧版add_axes_labels）
    axis_labels = ["X", "Y", "Z"]
    for i, label in enumerate(axis_labels):
        plotter.add_text(
            f"{label}",
            position=(bounds[2*i+1] + 0.5, bounds[1] if i==0 else bounds[3] if i==1 else bounds[5]),
            font_size=12,
            shadow=True,
            color="black"
        )
    return grid

def rendering_output(out):
    mask = np.where(out > 0.5, True, False)  # out: N
    points_masked = points[mask].astype(np.float32)
    # print(points.shape)
    point_cloud = pv.PolyData(points_masked)
    point_cloud['height'] = points_masked[:, 2]  
    cube = pv.Cube(x_length=1, y_length=1, z_length=1)
    mesh = point_cloud.glyph(scale=False, geom=cube, orient=False)
    
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        mesh, 
        scalars='height', 
        cmap='coolwarm', 
    )

    grid_bounds = [
        0, 16,
        0, 16,
        0, 16
    ]
    add_reference_grid(plotter, bounds=grid_bounds)
    plotter.add_floor(color='lightgray', opacity=0.1)  
    plotter.window_size = [800, 600]
    image = plotter.screenshot(return_img=True)
    plotter.close()
    pil_image = Image.fromarray(image)
    return pil_image


def get_color(opacity):
    if opacity >= 0:
        # 正值：红色到黄色渐变
        r = max(0, min(255, int(255 * opacity)))
        g = max(0, min(255, int(255 * (1 - opacity))))
        b = 0
    else:
        # 负值：蓝色到青色渐变
        r = 0
        g = max(0, min(255, int(255 * (1 + opacity))))
        b = max(0, min(255, int(255 * (1 + opacity))))
    return (r, g, b)


def rendering_gaussian(quaternions, centers, scales, gaussian_opacity):
    num_ellipsoids = centers.shape[0]
    
    # 1. 创建基础球体（用于变形为椭球）
    base_sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)  # 控制网格密度
    
    rotations = Rotation.from_quat(quaternions).as_matrix()
    # main_axes = rotations[:, :, 0]  # 取旋转矩阵的第一列（对应最大半轴方向）

    # 2. 将主轴方向映射到RGB颜色（-1~1 → 0~1）
    # 初始化颜色列表
    colors = []
    for opacity in gaussian_opacity:
        colors.append(get_color(opacity))

    ellipsoids = pv.MultiBlock()
    for i in range(num_ellipsoids):
        sphere = base_sphere.copy()
        transform = np.eye(4)
        transform[:3, :3] = rotations[i] * scales[i]  # 旋转+缩放
        transform[:3, 3] = centers[i]                # 平移
        sphere.transform(transform, inplace=True)
        sphere.cell_data["colors"] = np.tile(colors[i], (sphere.n_cells, 1))  # 每个面片着色
        ellipsoids.append(sphere)

    merged = ellipsoids.combine()  # 合并为一个网格（提升渲染性能）

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        merged,
        scalars="colors",
        rgb=True,               # 使用RGB颜色
        specular=0.8,           # 高光强度（增强立体感）
        smooth_shading=True,    # 平滑着色
        metallic=0.3,           # 金属质感（替代旧版lighting_params）
        roughness=0.5,          # 表面粗糙度（0-1）
        diffuse=0.8,            # 漫反射强度
        edge_color="white",     # 边缘高亮
        line_width=0.1,         # 边缘线宽
        opacity=0.7,            # 透明度
    )

    grid_bounds = [
        0, 16,
        0, 16,
        0, 16
    ]
    add_reference_grid(plotter, bounds=grid_bounds)

    plotter.enable_ssao(radius=0.1) 
    plotter.add_floor(color='lightgray', opacity=0.1) 
    plotter.window_size = [800, 600]
    image = plotter.screenshot(return_img=True)
    plotter.close()
    pil_image = Image.fromarray(image)
    return pil_image

frames_out = []
frames_gaussian = []  

# norm opacity
max_opacity = 0.
for data in vis_data:
    max_opacity = max(max_opacity, np.abs(data["opacity"]).max())

for data in vis_data:
    frames_out.append(rendering_output(data["out"]))
    
    # 只有非常少的高斯点起作用， 9 个是正数， 只有两个的scale 保持较大值
    
    # 预处理数据
    data["xyz"][:, 0] =  0.5 * cube_x * data["xyz"][:, 0] + 0.5 * cube_x
    data["xyz"][:, 1] =  0.5 * cube_y * data["xyz"][:, 1] + 0.5 * cube_y
    data["xyz"][:, 2] =  0.5 * cube_z * data["xyz"][:, 2] + 0.5 * cube_z
    
    data["scale"] = data["scale"] * 3
    
    data["opacity"] =  data["opacity"] / max_opacity

    mask = np.where(data["opacity"] > -1e9, True, False).squeeze(1)  # out: N
    
    frames_gaussian.append(rendering_gaussian(data["quat"][mask], data["xyz"][mask], data["scale"][mask], data["opacity"][mask]))
    

frames_out[0].save(
                "samples/training.gif",
                save_all=True,
                append_images=frames_out[1:],
                optimize=False,
                duration=100,
                loop=0,
            )

frames_gaussian[0].save(
                "samples/gaussian.gif",
                save_all=True,
                append_images=frames_gaussian[1:],
                optimize=False,
                duration=100,
                loop=0,
            )
