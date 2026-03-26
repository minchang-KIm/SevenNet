#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "swift_tp_sh.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kMetaCols = 9;
constexpr int kMaxWarpsPerBlock = 8;

enum PathMetaIndex : int {
  kXOffset = 0,
  kShOffset = 1,
  kOutOffset = 2,
  kWeightOffset = 3,
  kMul = 4,
  kDimIn = 5,
  kDimSh = 6,
  kDimOut = 7,
  kCoeffOffset = 8,
};

inline void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& edge_vec,
    const torch::Tensor& weight,
    const torch::Tensor& row_ptr,
    const torch::Tensor& col_idx,
    const torch::Tensor& path_meta,
    const torch::Tensor& cg_coeff,
    int64_t warps_per_block) {
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(
      edge_vec.scalar_type() == torch::kFloat32,
      "edge_vec must be float32");
  TORCH_CHECK(
      weight.scalar_type() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(
      row_ptr.scalar_type() == torch::kInt32, "row_ptr must be int32");
  TORCH_CHECK(
      col_idx.scalar_type() == torch::kInt32, "col_idx must be int32");
  TORCH_CHECK(
      path_meta.scalar_type() == torch::kInt32, "path_meta must be int32");
  TORCH_CHECK(
      cg_coeff.scalar_type() == torch::kFloat32, "cg_coeff must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(edge_vec.dim() == 2 && edge_vec.size(1) == 3, "edge_vec shape");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(path_meta.dim() == 2 && path_meta.size(1) == kMetaCols, "meta");
  TORCH_CHECK(
      warps_per_block >= 1 && warps_per_block <= kMaxWarpsPerBlock,
      "warps_per_block out of range");
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float contract_component(
    const float* x_ptr,
    const float* sh_ptr,
    const float* coeff_ptr,
    int dim_in,
    int dim_sh,
    int mout) {
  float value = 0.0f;
  const float* coeff_m = coeff_ptr + mout * dim_in * dim_sh;
  for (int m1 = 0; m1 < dim_in; ++m1) {
    const float x_value = x_ptr[m1];
    const float* coeff_row = coeff_m + m1 * dim_sh;
    for (int m2 = 0; m2 < dim_sh; ++m2) {
      value = fmaf(coeff_row[m2], x_value * sh_ptr[m2], value);
    }
  }
  return value;
}

__device__ void process_row_warp(
    int row,
    int lane,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const float* edge_vec,
    const float* weight,
    int weight_stride,
    const float* x,
    int x_stride,
    const int32_t* path_meta,
    int num_paths,
    const float* cg_coeff,
    float* out,
    int out_stride,
    bool enable_scalar_fastpath) {
  const int row_start = row_ptr[row];
  const int row_end = row_ptr[row + 1];
  if (row_start == row_end) {
    return;
  }

  for (int path = 0; path < num_paths; ++path) {
    const int32_t* meta = path_meta + path * kMetaCols;
    const int x_offset = meta[kXOffset];
    const int sh_offset = meta[kShOffset];
    const int out_offset = meta[kOutOffset];
    const int weight_offset = meta[kWeightOffset];
    const int mul = meta[kMul];
    const int dim_in = meta[kDimIn];
    const int dim_sh = meta[kDimSh];
    const int dim_out = meta[kDimOut];
    const int coeff_offset = meta[kCoeffOffset];
    const float* coeff_base = cg_coeff + coeff_offset;

    for (int u = 0; u < mul; ++u) {
      if (enable_scalar_fastpath && dim_out == 1) {
        float accum = 0.0f;
        for (int edge = row_start + lane; edge < row_end; edge += kWarpSize) {
          const int src = col_idx[edge];
          const float* edge_ptr = edge_vec + edge * 3;
          const float radius = sqrtf(
              edge_ptr[0] * edge_ptr[0] +
              edge_ptr[1] * edge_ptr[1] +
              edge_ptr[2] * edge_ptr[2]);
          float sh[16];
          compute_real_sh_l3_cartesian(
              edge_ptr[0], edge_ptr[1], edge_ptr[2], radius, sh);
          const float weight_value = weight[edge * weight_stride + weight_offset + u];
          const float* x_ptr = x + src * x_stride + x_offset + u * dim_in;
          accum += weight_value * contract_component(
              x_ptr, sh + sh_offset, coeff_base, dim_in, dim_sh, 0);
        }
        accum = warp_sum(accum);
        if (lane == 0) {
          out[row * out_stride + out_offset + u] = accum;
        }
        continue;
      }

      for (int mout = 0; mout < dim_out; ++mout) {
        float accum = 0.0f;
        for (int edge = row_start + lane; edge < row_end; edge += kWarpSize) {
          const int src = col_idx[edge];
          const float* edge_ptr = edge_vec + edge * 3;
          const float radius = sqrtf(
              edge_ptr[0] * edge_ptr[0] +
              edge_ptr[1] * edge_ptr[1] +
              edge_ptr[2] * edge_ptr[2]);
          float sh[16];
          compute_real_sh_l3_cartesian(
              edge_ptr[0], edge_ptr[1], edge_ptr[2], radius, sh);
          const float weight_value = weight[edge * weight_stride + weight_offset + u];
          const float* x_ptr = x + src * x_stride + x_offset + u * dim_in;
          accum += weight_value * contract_component(
              x_ptr, sh + sh_offset, coeff_base, dim_in, dim_sh, mout);
        }
        accum = warp_sum(accum);
        if (lane == 0) {
          out[row * out_stride + out_offset + u * dim_out + mout] = accum;
        }
      }
    }
  }
}

__device__ void process_row_cta(
    int row,
    int lane,
    int warp_id,
    int num_warps,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const float* edge_vec,
    const float* weight,
    int weight_stride,
    const float* x,
    int x_stride,
    const int32_t* path_meta,
    int num_paths,
    const float* cg_coeff,
    float* warp_sums,
    float* out,
    int out_stride,
    bool enable_scalar_fastpath) {
  const int row_start = row_ptr[row];
  const int row_end = row_ptr[row + 1];
  if (row_start == row_end) {
    return;
  }

  for (int path = 0; path < num_paths; ++path) {
    const int32_t* meta = path_meta + path * kMetaCols;
    const int x_offset = meta[kXOffset];
    const int sh_offset = meta[kShOffset];
    const int out_offset = meta[kOutOffset];
    const int weight_offset = meta[kWeightOffset];
    const int mul = meta[kMul];
    const int dim_in = meta[kDimIn];
    const int dim_sh = meta[kDimSh];
    const int dim_out = meta[kDimOut];
    const int coeff_offset = meta[kCoeffOffset];
    const float* coeff_base = cg_coeff + coeff_offset;

    for (int u = 0; u < mul; ++u) {
      if (enable_scalar_fastpath && dim_out == 1) {
        float local = 0.0f;
        for (int edge = row_start + threadIdx.x; edge < row_end;
             edge += blockDim.x) {
          const int src = col_idx[edge];
          const float* edge_ptr = edge_vec + edge * 3;
          const float radius = sqrtf(
              edge_ptr[0] * edge_ptr[0] +
              edge_ptr[1] * edge_ptr[1] +
              edge_ptr[2] * edge_ptr[2]);
          float sh[16];
          compute_real_sh_l3_cartesian(
              edge_ptr[0], edge_ptr[1], edge_ptr[2], radius, sh);
          const float weight_value = weight[edge * weight_stride + weight_offset + u];
          const float* x_ptr = x + src * x_stride + x_offset + u * dim_in;
          local += weight_value * contract_component(
              x_ptr, sh + sh_offset, coeff_base, dim_in, dim_sh, 0);
        }
        local = warp_sum(local);
        if (lane == 0) {
          warp_sums[warp_id] = local;
        }
        __syncthreads();
        if (warp_id == 0) {
          float block_value = lane < num_warps ? warp_sums[lane] : 0.0f;
          block_value = warp_sum(block_value);
          if (lane == 0) {
            out[row * out_stride + out_offset + u] = block_value;
          }
        }
        __syncthreads();
        continue;
      }

      for (int mout = 0; mout < dim_out; ++mout) {
        float local = 0.0f;
        for (int edge = row_start + threadIdx.x; edge < row_end;
             edge += blockDim.x) {
          const int src = col_idx[edge];
          const float* edge_ptr = edge_vec + edge * 3;
          const float radius = sqrtf(
              edge_ptr[0] * edge_ptr[0] +
              edge_ptr[1] * edge_ptr[1] +
              edge_ptr[2] * edge_ptr[2]);
          float sh[16];
          compute_real_sh_l3_cartesian(
              edge_ptr[0], edge_ptr[1], edge_ptr[2], radius, sh);
          const float weight_value = weight[edge * weight_stride + weight_offset + u];
          const float* x_ptr = x + src * x_stride + x_offset + u * dim_in;
          local += weight_value * contract_component(
              x_ptr, sh + sh_offset, coeff_base, dim_in, dim_sh, mout);
        }
        local = warp_sum(local);
        if (lane == 0) {
          warp_sums[warp_id] = local;
        }
        __syncthreads();
        if (warp_id == 0) {
          float block_value = lane < num_warps ? warp_sums[lane] : 0.0f;
          block_value = warp_sum(block_value);
          if (lane == 0) {
            out[row * out_stride + out_offset + u * dim_out + mout] =
                block_value;
          }
        }
        __syncthreads();
      }
    }
  }
}

__global__ void swift_kernel_packed_small_rows(
    const int32_t* row_ids,
    int num_rows,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const float* edge_vec,
    const float* weight,
    int weight_stride,
    const float* x,
    int x_stride,
    const int32_t* path_meta,
    int num_paths,
    const float* cg_coeff,
    float* out,
    int out_stride,
    bool enable_scalar_fastpath) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int num_warps = blockDim.x / kWarpSize;
  const int row_index = blockIdx.x * num_warps + warp_id;
  if (row_index >= num_rows) {
    return;
  }
  process_row_warp(
      row_ids[row_index],
      lane,
      row_ptr,
      col_idx,
      edge_vec,
      weight,
      weight_stride,
      x,
      x_stride,
      path_meta,
      num_paths,
      cg_coeff,
      out,
      out_stride,
      enable_scalar_fastpath);
}

__global__ void swift_kernel_warp_per_row(
    const int32_t* row_ids,
    int num_rows,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const float* edge_vec,
    const float* weight,
    int weight_stride,
    const float* x,
    int x_stride,
    const int32_t* path_meta,
    int num_paths,
    const float* cg_coeff,
    float* out,
    int out_stride,
    bool enable_scalar_fastpath) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int num_warps = blockDim.x / kWarpSize;
  const int row_index = blockIdx.x * num_warps + warp_id;
  if (row_index >= num_rows) {
    return;
  }
  process_row_warp(
      row_ids[row_index],
      lane,
      row_ptr,
      col_idx,
      edge_vec,
      weight,
      weight_stride,
      x,
      x_stride,
      path_meta,
      num_paths,
      cg_coeff,
      out,
      out_stride,
      enable_scalar_fastpath);
}

__global__ void swift_kernel_cta_per_row(
    const int32_t* row_ids,
    int num_rows,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const float* edge_vec,
    const float* weight,
    int weight_stride,
    const float* x,
    int x_stride,
    const int32_t* path_meta,
    int num_paths,
    const float* cg_coeff,
    float* out,
    int out_stride,
    bool enable_scalar_fastpath) {
  const int row_index = blockIdx.x;
  if (row_index >= num_rows) {
    return;
  }
  __shared__ float warp_sums[kMaxWarpsPerBlock];
  const int lane = threadIdx.x % kWarpSize;
  const int warp_id = threadIdx.x / kWarpSize;
  const int num_warps = blockDim.x / kWarpSize;
  process_row_cta(
      row_ids[row_index],
      lane,
      warp_id,
      num_warps,
      row_ptr,
      col_idx,
      edge_vec,
      weight,
      weight_stride,
      x,
      x_stride,
      path_meta,
      num_paths,
      cg_coeff,
      warp_sums,
      out,
      out_stride,
      enable_scalar_fastpath);
}

}  // namespace

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
    bool enable_scalar_fastpath) {
  check_inputs(
      x,
      edge_vec,
      weight,
      row_ptr,
      col_idx,
      path_meta,
      cg_coeff,
      warps_per_block);

  (void)small_row_threshold;
  (void)cta_row_threshold;
  (void)out_tile;

  auto out = torch::zeros(
      {x.size(0), out_dim},
      torch::TensorOptions().dtype(x.dtype()).device(x.device()));

  if (edge_vec.size(0) == 0) {
    return out;
  }

  const dim3 block(static_cast<unsigned int>(warps_per_block * kWarpSize));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (small_rows.numel() > 0) {
    const int num_rows = static_cast<int>(small_rows.size(0));
    const dim3 grid(
        static_cast<unsigned int>((num_rows + warps_per_block - 1) /
                                  warps_per_block));
    swift_kernel_packed_small_rows<<<grid, block, 0, stream>>>(
        small_rows.data_ptr<int32_t>(),
        num_rows,
        row_ptr.data_ptr<int32_t>(),
        col_idx.data_ptr<int32_t>(),
        edge_vec.data_ptr<float>(),
        weight.data_ptr<float>(),
        static_cast<int>(weight.size(1)),
        x.data_ptr<float>(),
        static_cast<int>(x.size(1)),
        path_meta.data_ptr<int32_t>(),
        static_cast<int>(path_meta.size(0)),
        cg_coeff.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(out.size(1)),
        enable_scalar_fastpath);
  }

  if (medium_rows.numel() > 0) {
    const int num_rows = static_cast<int>(medium_rows.size(0));
    const dim3 grid(
        static_cast<unsigned int>((num_rows + warps_per_block - 1) /
                                  warps_per_block));
    swift_kernel_warp_per_row<<<grid, block, 0, stream>>>(
        medium_rows.data_ptr<int32_t>(),
        num_rows,
        row_ptr.data_ptr<int32_t>(),
        col_idx.data_ptr<int32_t>(),
        edge_vec.data_ptr<float>(),
        weight.data_ptr<float>(),
        static_cast<int>(weight.size(1)),
        x.data_ptr<float>(),
        static_cast<int>(x.size(1)),
        path_meta.data_ptr<int32_t>(),
        static_cast<int>(path_meta.size(0)),
        cg_coeff.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(out.size(1)),
        enable_scalar_fastpath);
  }

  if (high_rows.numel() > 0) {
    const int num_rows = static_cast<int>(high_rows.size(0));
    const dim3 grid(static_cast<unsigned int>(num_rows));
    swift_kernel_cta_per_row<<<grid, block, 0, stream>>>(
        high_rows.data_ptr<int32_t>(),
        num_rows,
        row_ptr.data_ptr<int32_t>(),
        col_idx.data_ptr<int32_t>(),
        edge_vec.data_ptr<float>(),
        weight.data_ptr<float>(),
        static_cast<int>(weight.size(1)),
        x.data_ptr<float>(),
        static_cast<int>(x.size(1)),
        path_meta.data_ptr<int32_t>(),
        static_cast<int>(path_meta.size(0)),
        cg_coeff.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(out.size(1)),
        enable_scalar_fastpath);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
