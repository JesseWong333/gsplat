#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  
    m.def("nd_rasterize_sum_forward", &nd_rasterize_forward_sum_tensor);
    m.def("nd_rasterize_sum_backward", &nd_rasterize_backward_sum_tensor);

    m.def("project_gaussians_forward", &project_gaussians_forward_tensor);
    m.def("project_gaussians_backward", &project_gaussians_backward_tensor);
   
    // utils
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);
    m.def("get_tile_bin_edges_pts", &get_tile_bin_edges_pts_tensor);
}
