import mcubes
import numpy as np

import open3d.core as o3c
import trimesh
import time
def create_f_function(hashmap, block_size):
    def f(x, y, z):
        points = np.array([x, y, z])
        point_block_coords = np.floor(points / block_size)
        points_relative = points - point_block_coords * block_size
        query_keys = o3c.Tensor(point_block_coords.astype(np.int64)[None, :],
                            device=o3c.Device('cpu:0'))
        buf_indices, masks = hashmap.find(query_keys)
        
        if not masks[0]:
            return 0.
        # valid_keys = query_keys[masks]
        buf_indices = buf_indices[masks].to(o3c.int64)
        valid_vals = hashmap.value_tensor()[buf_indices] # 1* 16*16*16
        occ =  valid_vals[0, int(points_relative[0]), int(points_relative[1]), int(points_relative[2])]
        return occ
    return f

if __name__ == "__main__":
    bounds=[0, 0, 0, 1000, 1000, 1000]
    voxel_size = 1
    block_resolution = 16

    hashmap = o3c.HashMap.load("./samples/sparse_voxel_gaussian_rendering.npz")
    block_coords = hashmap.key_tensor().numpy() 
    block_semantics = hashmap.value_tensor().numpy()

    block_size = voxel_size * block_resolution
    block_min = block_coords * block_size + np.array([bounds[0], bounds[1], bounds[2]]) # N, 3
    nonzero_indices = np.where(block_semantics > 0)
    valid_semantics = block_semantics[nonzero_indices] 

    start_time = time.time()
    
    vertices, triangles = mcubes.marching_cubes_func((0, 0, 0), (1000, 1000, 1000), 1000, 1000, 1000, create_f_function(hashmap, block_size), 0.5)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
     
    print("Time: ", time.time() - start_time)
    mesh.export("./samples/output_mesh.ply")
    