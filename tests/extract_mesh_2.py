import mcubes
import numpy as np
import open3d as o3d
import open3d.core as o3c
import trimesh
import time

# 避免 marching cube锯齿状的做法，是不是应该用logistic值？还是不够光滑， label smothing也没用
bounds=[0, 0, 0, 1000, 1000, 1000]
voxel_size = 1
block_resolution = 16

hashmap = o3c.HashMap.load("./samples/sparse_voxel_gaussian_rendering.npz")

cube = np.zeros((1000, 1000, 1000))

block_coords = hashmap.key_tensor().numpy() # N, 3
block_semantics = hashmap.value_tensor().numpy() # N, 16, 16, 16
# block_semantics[block_semantics > 0] = 1

block_size = voxel_size * block_resolution
block_mins = block_coords * block_size + np.array([bounds[0], bounds[1], bounds[2]]) # N, 3

for i, block_min in enumerate(block_mins):
    cube[block_min[0]:block_min[0]+block_size, block_min[1]:block_min[1]+block_size, block_min[2]:block_min[2]+block_size] = block_semantics[i]

start_time = time.time()
vertices, triangles = mcubes.marching_cubes(cube, 0.5)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()
mesh_lap = mesh.filter_smooth_laplacian(number_of_iterations=5, lambda_filter=0.5)
mesh_lap.compute_vertex_normals()

o3d.io.write_triangle_mesh("./samples/output_mesh.ply", mesh_lap)
print("Time: ", time.time() - start_time)

