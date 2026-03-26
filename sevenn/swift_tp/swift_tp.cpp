#include <torch/extension.h>

torch::Tensor swift_forward_cuda(
    torch::Tensor x,
    torch::Tensor edge_vec,
    torch::Tensor weight,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor small_rows,
    torch::Tensor medium_rows,
    torch::Tensor high_rows,
    torch::Tensor path_meta,
    torch::Tensor cg_coeff,
    int64_t out_dim,
    int64_t small_row_threshold,
    int64_t cta_row_threshold,
    int64_t out_tile,
    int64_t warps_per_block,
    bool enable_scalar_fastpath);

torch::Tensor swift_forward(
    torch::Tensor x,
    torch::Tensor edge_vec,
    torch::Tensor weight,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor small_rows,
    torch::Tensor medium_rows,
    torch::Tensor high_rows,
    torch::Tensor path_meta,
    torch::Tensor cg_coeff,
    int64_t out_dim,
    int64_t small_row_threshold,
    int64_t cta_row_threshold,
    int64_t out_tile,
    int64_t warps_per_block,
    bool enable_scalar_fastpath) {
  TORCH_CHECK(x.is_cuda(), "SWIFT-TP expects CUDA tensors");
  TORCH_CHECK(edge_vec.is_cuda(), "SWIFT-TP expects CUDA tensors");
  TORCH_CHECK(weight.is_cuda(), "SWIFT-TP expects CUDA tensors");
  TORCH_CHECK(row_ptr.is_cuda(), "SWIFT-TP expects CUDA tensors");
  TORCH_CHECK(col_idx.is_cuda(), "SWIFT-TP expects CUDA tensors");
  TORCH_CHECK(path_meta.is_cuda(), "SWIFT-TP expects CUDA tensors");
  TORCH_CHECK(cg_coeff.is_cuda(), "SWIFT-TP expects CUDA tensors");
  return swift_forward_cuda(
      x,
      edge_vec,
      weight,
      row_ptr,
      col_idx,
      small_rows,
      medium_rows,
      high_rows,
      path_meta,
      cg_coeff,
      out_dim,
      small_row_threshold,
      cta_row_threshold,
      out_tile,
      warps_per_block,
      enable_scalar_fastpath);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swift_forward", &swift_forward, "SWIFT-TP forward");
}
