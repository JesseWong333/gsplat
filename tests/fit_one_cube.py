# Gaussianformer v2 类似的方法中

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
    
    def project(self, ):
        xys, depths, radii, conics, num_tiles_hit = project_gaussians(self.get_xyz, self.get_scaling, 1, 
                                                                                       self.get_rotation, self.H, self.W, self.L,
                                                                                            self.tile_bounds)
        return xys, depths, radii, conics, num_tiles_hit
    
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
    
    # ------------------------------------------------------------------------------------------------------------------
    # 测试cube拟合
    cube_x = 100
    cube_y = 100
    cube_z = 100
    
    gt_image = torch.zeros((cube_x, cube_y, cube_z))
    # make top left and bottom right red, blue
    gt_image[: cube_x // 2, : cube_y // 2, cube_z // 2 :] = torch.tensor([1.0])
    gt_image[cube_x // 2 :, cube_y // 2 :, :cube_z // 2] = torch.tensor([1.0])
        
    # normlize
    mins = [0., 0., 0.]
    
    grid_size = 6
    
    num_channel = 2  # 占据或者不占据
    num_points = 200 # 高斯点
    
    # 我这样做其实是一个生成式的3D模型，雷达是采样的点
    gaussian_model = GaussianSSC(num_points=num_points, H = cube_x, W = cube_y, L = cube_z, BLOCK_W = grid_size, BLOCK_H = grid_size, BLOCK_L = grid_size, lidar_mins=mins).cuda()
    
    steps = 4000

    # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.8, 1]).cuda())
    # loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([1.])).cuda()
    loss_fn = nn.BCELoss(weight=torch.tensor([1.])).cuda()

    optimizer = torch.optim.Adam(gaussian_model.parameters(), lr=0.05)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    gaussian_model.train()

    # --------------------------------------------------------------
    start = 0
    end = 100
    step = 1
    x = np.arange(start, end, step)
    y = np.arange(start, end, step)
    z = np.arange(start, end, step)

    # Generate all combinations of coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    query_points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    target = gt_image[query_points[:, 0], query_points[:, 1], query_points[:, 2]]
    # --------------------------------------------------------------

    progress_bar = tqdm(range(steps), desc="Training")
    
    vis_data = []
    for i in progress_bar:
        
        if i == 900:
            i = 901
            pass
        sampled_points = torch.from_numpy(query_points).cuda().float()
        sampled_target = target.cuda().float()

        out = gaussian_model.forward(sampled_points) # N * 1; 渲染就是全 0是初始化的问题
        # out = out.squeeze(1)
        loss = loss_fn(out, sampled_target)
        # torch.nn.utils.clip_grad_norm_(gaussian_model.parameters(), max_norm=1.0)
        loss.backward()   
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_description(f"Step {i}, Loss: {loss.item()}")
        if i % 100 == 0:
            vis_data.append(
                {
                    "out": out.cpu().detach().numpy(),     
                    "xyz": gaussian_model.get_xyz.cpu().detach().numpy(),
                    "scale": gaussian_model.get_scaling.cpu().detach().numpy(),
                    "quat": gaussian_model.get_rotation.cpu().detach().numpy(),
                    "opacity": gaussian_model.get_opacity.cpu().detach().numpy(),
                }       
                # out.sigmoid().cpu().de    tach().numpy() 
            )

# 结果分析
# xys, depths, radii, conics, num_tiles_hit  = gaussian_model.project()
# upper_coner_mask = (xys[:, 0] < 50) & (xys[:, 1] < 50) & (xys[:, 2] > 50)
# radii = radii[upper_coner_mask]

# mask = np.where(out > 0.5, True, False)  

pass


# visulize
# -------------------------------------------------------------------------------------------------------
# def rendering_output(out):
#     mask = np.where(out > 0.5, True, False)  # out: N
#     points_masked = points[mask].astype(np.float32)
#     # print(points.shape)
#     point_cloud = pv.PolyData(points_masked)
#     point_cloud['height'] = points_masked[:, 2]  
#     cube = pv.Cube(x_length=1, y_length=1, z_length=1)
#     mesh = point_cloud.glyph(scale=False, geom=cube, orient=False)
    
#     plotter = pv.Plotter(off_screen=True)
#     plotter.add_mesh(
#         mesh, 
#         scalars='height', 
#         cmap='coolwarm', 
#     )

#     plotter.window_size = [800, 600]
#     image = plotter.screenshot(return_img=True)
#     plotter.close()
#     pil_image = Image.fromarray(image)
#     return pil_image

def rendering_output(out, points, thresh=0.3):
    # 创建可视化掩码：只显示值大于0.01的点
    mask = (out > thresh)
    points_masked = points[mask].astype(np.float32)
    values_masked = out[mask]  # 对应的out值
    
    # 创建点云对象
    point_cloud = pv.PolyData(points_masked)
    
    # 将out值作为标量数据添加到点云
    point_cloud['intensity'] = values_masked
    
    # 创建立方体作为每个点的glyph
    cube = pv.Cube(x_length=1, y_length=1, z_length=1)
    
    # 创建glyph网格 - 每个点变成一个立方体
    mesh = point_cloud.glyph(scale=False, geom=cube, orient=False)
    
    # 创建绘图器
    plotter = pv.Plotter(off_screen=True)
    
    # 添加网格到绘图器，使用coolwarm渐变色
    plotter.add_mesh(
        mesh, 
        scalars='intensity',  # 使用out值作为颜色标量
        cmap='coolwarm',      # 使用渐变色
        clim=[0.001, 1.0],    # 设置颜色范围，最小值设为0.001
        show_scalar_bar=True  # 显示颜色条
    )
    
    # 设置窗口大小
    plotter.window_size = [800, 600]
    
    # 截图并转换为PIL图像
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
    plotter.enable_ssao(radius=0.1) 
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
    frames_out.append(rendering_output(data["out"], query_points))
    
    # 预处理数据
    data["xyz"][:, 0] =  0.5 * cube_x * data["xyz"][:, 0] + 0.5 * cube_x
    data["xyz"][:, 1] =  0.5 * cube_y * data["xyz"][:, 1] + 0.5 * cube_y
    data["xyz"][:, 2] =  0.5 * cube_z * data["xyz"][:, 2] + 0.5 * cube_z
    
    data["scale"] = data["scale"] * 3
    
    data["opacity"] =  data["opacity"] / max_opacity
    
    frames_gaussian.append(rendering_gaussian(data["quat"], data["xyz"], data["scale"], data["opacity"]))
    

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
