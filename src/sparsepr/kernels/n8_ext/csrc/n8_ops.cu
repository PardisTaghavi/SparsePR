#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void wan_qkv_norm_rope_layout_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ q_weight,
    const scalar_t* __restrict__ k_weight,
    const float* __restrict__ freq_real,
    const float* __restrict__ freq_imag,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int64_t rows,
    int64_t tokens,
    int64_t channels,
    int64_t heads,
    int64_t head_dim,
    float epsilon) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  extern __shared__ float shared[];
  float q_sum = 0.0f;
  float k_sum = 0.0f;
  const int64_t input_base = row * channels;
  for (int64_t c = tid; c < channels; c += blockDim.x) {
    const float qv = static_cast<float>(q[input_base + c]);
    const float kv = static_cast<float>(k[input_base + c]);
    q_sum += qv * qv;
    k_sum += kv * kv;
  }
  shared[tid] = q_sum;
  shared[blockDim.x + tid] = k_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
      shared[blockDim.x + tid] += shared[blockDim.x + tid + stride];
    }
    __syncthreads();
  }
  const float q_inv = rsqrtf(shared[0] / static_cast<float>(channels) + epsilon);
  const float k_inv = rsqrtf(shared[blockDim.x] / static_cast<float>(channels) + epsilon);
  const int64_t batch = row / tokens;
  const int64_t token = row - batch * tokens;

  for (int64_t c = tid; c < channels; c += blockDim.x) {
    const int64_t head = c / head_dim;
    const int64_t d = c - head * head_dim;
    const int64_t output_offset =
        ((batch * heads + head) * tokens + token) * head_dim + d;
    v_out[output_offset] = v[input_base + c];
  }

  const int64_t complex_channels = channels / 2;
  for (int64_t pair = tid; pair < complex_channels; pair += blockDim.x) {
    const int64_t head = pair / (head_dim / 2);
    const int64_t pair_in_head = pair - head * (head_dim / 2);
    const int64_t d0 = pair_in_head * 2;
    const int64_t c0 = head * head_dim + d0;
    const int64_t c1 = c0 + 1;
    // Wan rounds RMSNorm back to the model dtype before applying RoPE.
    const scalar_t q0_round = static_cast<scalar_t>(
        static_cast<float>(q[input_base + c0]) * q_inv *
        static_cast<float>(q_weight[c0]));
    const scalar_t q1_round = static_cast<scalar_t>(
        static_cast<float>(q[input_base + c1]) * q_inv *
        static_cast<float>(q_weight[c1]));
    const scalar_t k0_round = static_cast<scalar_t>(
        static_cast<float>(k[input_base + c0]) * k_inv *
        static_cast<float>(k_weight[c0]));
    const scalar_t k1_round = static_cast<scalar_t>(
        static_cast<float>(k[input_base + c1]) * k_inv *
        static_cast<float>(k_weight[c1]));
    const float fr = freq_real[token * (head_dim / 2) + pair_in_head];
    const float fi = freq_imag[token * (head_dim / 2) + pair_in_head];
    const float q0 = static_cast<float>(q0_round);
    const float q1 = static_cast<float>(q1_round);
    const float k0 = static_cast<float>(k0_round);
    const float k1 = static_cast<float>(k1_round);
    const int64_t output_base =
        ((batch * heads + head) * tokens + token) * head_dim + d0;
    q_out[output_base] = static_cast<scalar_t>(q0 * fr - q1 * fi);
    q_out[output_base + 1] = static_cast<scalar_t>(q0 * fi + q1 * fr);
    k_out[output_base] = static_cast<scalar_t>(k0 * fr - k1 * fi);
    k_out[output_base + 1] = static_cast<scalar_t>(k0 * fi + k1 * fr);
  }
}

template <typename scalar_t>
__global__ void cosmos25_qkv_norm_rope_bshd_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ rope_angles,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int64_t rows,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    float epsilon) {
  // One block owns one [batch, token, head] row. Cosmos normalizes Q and K
  // independently over head_dim, unlike native Wan's full-channel norm.
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  extern __shared__ float shared[];
  float q_sum = 0.0f;
  float k_sum = 0.0f;
  const int64_t input_base = row * head_dim;
  for (int64_t d = tid; d < head_dim; d += blockDim.x) {
    const float qv = static_cast<float>(q[input_base + d]);
    const float kv = static_cast<float>(k[input_base + d]);
    q_sum += qv * qv;
    k_sum += kv * kv;
  }
  shared[tid] = q_sum;
  shared[blockDim.x + tid] = k_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
      shared[blockDim.x + tid] += shared[blockDim.x + tid + stride];
    }
    __syncthreads();
  }

  const float q_inv =
      rsqrtf(shared[0] / static_cast<float>(head_dim) + epsilon);
  const float k_inv =
      rsqrtf(shared[blockDim.x] / static_cast<float>(head_dim) + epsilon);
  const int64_t token_head = row % (tokens * heads);
  const int64_t token = token_head / heads;
  const int64_t half_dim = head_dim / 2;

  for (int64_t d = tid; d < head_dim; d += blockDim.x) {
    const int64_t pair_d = d < half_dim ? d + half_dim : d - half_dim;
    const float pair_sign = d < half_dim ? -1.0f : 1.0f;

    // Transformer Engine RMSNorm returns the model dtype before fused RoPE.
    // Preserve that rounding boundary before applying the FP32 angle.
    const scalar_t q_round = static_cast<scalar_t>(
        static_cast<float>(q[input_base + d]) * q_inv * q_weight[d]);
    const scalar_t q_pair_round = static_cast<scalar_t>(
        static_cast<float>(q[input_base + pair_d]) * q_inv *
        q_weight[pair_d]);
    const scalar_t k_round = static_cast<scalar_t>(
        static_cast<float>(k[input_base + d]) * k_inv * k_weight[d]);
    const scalar_t k_pair_round = static_cast<scalar_t>(
        static_cast<float>(k[input_base + pair_d]) * k_inv *
        k_weight[pair_d]);

    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(rope_angles[token * head_dim + d], &sine, &cosine);
    q_out[input_base + d] = static_cast<scalar_t>(
        static_cast<float>(q_round) * cosine +
        pair_sign * static_cast<float>(q_pair_round) * sine);
    k_out[input_base + d] = static_cast<scalar_t>(
        static_cast<float>(k_round) * cosine +
        pair_sign * static_cast<float>(k_pair_round) * sine);
    v_out[input_base + d] = v[input_base + d];
  }
}

__device__ __forceinline__ bool better_pair(
    float candidate_distance,
    int candidate_index,
    float current_distance,
    int current_index) {
  return candidate_distance < current_distance ||
      (candidate_distance == current_distance && candidate_index < current_index);
}

__global__ void role_cluster_assign_accumulate_q64_kernel(
    const float* __restrict__ features,
    const float* __restrict__ centroids,
    int32_t* __restrict__ labels,
    int32_t* __restrict__ sizes,
    float* __restrict__ sums,
    int64_t tokens,
    int clusters) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t bh = row / tokens;
  const int64_t token = row - bh * tokens;
  const int tid = threadIdx.x;
  __shared__ float best_distance[256];
  __shared__ int best_index[256];
  __shared__ int selected;

  float local_distance = std::numeric_limits<float>::max();
  int local_index = 0x7fffffff;
  const float* point = features + (bh * tokens + token) * 64;
  for (int cluster = tid; cluster < clusters; cluster += blockDim.x) {
    const float* center = centroids + (bh * clusters + cluster) * 64;
    float distance = 0.0f;
#pragma unroll
    for (int r = 0; r < 64; ++r) {
      const float delta = point[r] - center[r];
      distance = fmaf(delta, delta, distance);
    }
    if (better_pair(distance, cluster, local_distance, local_index)) {
      local_distance = distance;
      local_index = cluster;
    }
  }
  best_distance[tid] = local_distance;
  best_index[tid] = local_index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair(
            best_distance[tid + stride], best_index[tid + stride],
            best_distance[tid], best_index[tid])) {
      best_distance[tid] = best_distance[tid + stride];
      best_index[tid] = best_index[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    selected = best_index[0];
    labels[row] = selected;
    atomicAdd(sizes + bh * clusters + selected, 1);
  }
  __syncthreads();
  for (int r = tid; r < 64; r += blockDim.x) {
    atomicAdd(sums + (bh * clusters + selected) * 64 + r, point[r]);
  }
}

__global__ void role_cluster_finalize_q64_kernel(
    const float* __restrict__ sums,
    const int32_t* __restrict__ sizes,
    const float* __restrict__ previous,
    float* __restrict__ output,
    int clusters) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int r = threadIdx.x;
  if (r >= 64) {
    return;
  }
  const int64_t bh = row / clusters;
  const int cluster = row - bh * clusters;
  const int32_t count = sizes[bh * clusters + cluster];
  const int64_t offset = (bh * clusters + cluster) * 64 + r;
  output[offset] = count > 0 ? sums[offset] / static_cast<float>(count) : previous[offset];
}

__global__ void selector_budget_kernel(
    const int64_t* __restrict__ order,
    const int32_t* __restrict__ sizes,
    const int32_t* __restrict__ budgets,
    bool* __restrict__ output,
    int32_t* __restrict__ selected_ids,
    int32_t* __restrict__ selected_counts,
    int q_clusters,
    int k_clusters) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t bh = row / q_clusters;
  const int q = row - bh * q_clusters;
  if (threadIdx.x != 0) {
    return;
  }
  const int64_t stride = static_cast<int64_t>(k_clusters) + 1;
  bool* output_row = output + ((bh * (q_clusters + 1) + q + 1) * stride);
  output_row[0] = true;
  int selected_cost = 0;
  int first_crossing = -1;
  const int budget = budgets[row];
  for (int position = 0; position < k_clusters; ++position) {
    const int cluster = static_cast<int>(order[row * k_clusters + position]);
    const int cost = sizes[bh * k_clusters + cluster];
    if (cost <= 0) {
      continue;
    }
    if (first_crossing < 0 && selected_cost + cost <= budget) {
      output_row[cluster + 1] = true;
      selected_cost += cost;
    } else if (first_crossing < 0) {
      first_crossing = cluster;
    }
  }
  if (budget > 0 && selected_cost < budget && first_crossing >= 0) {
    output_row[first_crossing + 1] = true;
  }
  if (selected_ids != nullptr && selected_counts != nullptr) {
    int count = 0;
    int32_t* ids_row = selected_ids + row * k_clusters;
    for (int cluster = 0; cluster < k_clusters; ++cluster) {
      if (output_row[cluster + 1]) {
        ids_row[count++] = cluster;
      }
    }
    selected_counts[row] = count;
  }
  if (q == 0) {
    bool* header = output + bh * (q_clusters + 1) * stride;
    header[0] = true;
  }
}

__global__ void compact_selected_clusters_kernel(
    const bool* __restrict__ selected_mask,
    int32_t* __restrict__ selected_ids,
    int32_t* __restrict__ selected_counts,
    int rows,
    int k_clusters) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  if (row >= rows || lane >= 32) {
    return;
  }

  const bool* mask_row =
      selected_mask + static_cast<int64_t>(row) * k_clusters;
  int32_t* ids_row =
      selected_ids + static_cast<int64_t>(row) * k_clusters;
  int write_base = 0;
  for (int base = 0; base < k_clusters; base += 32) {
    const int cluster = base + lane;
    const bool keep = cluster < k_clusters && mask_row[cluster];
    const unsigned int keep_bits = __ballot_sync(0xffffffffu, keep);
    const unsigned int lower_bits =
        lane == 0 ? 0u : keep_bits & ((1u << lane) - 1u);
    if (keep) {
      ids_row[write_base + __popc(lower_bits)] = cluster;
    }
    write_base += __popc(keep_bits);
  }
  if (lane == 0) {
    selected_counts[row] = write_base;
  }
}

std::vector<torch::Tensor> wan_qkv_norm_rope_layout(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor freq_real,
    torch::Tensor freq_imag,
    int64_t heads,
    double epsilon) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Wan QKV inputs must be CUDA tensors");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Wan QKV inputs must be contiguous");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes() && q.dim() == 3,
              "Wan QKV inputs must have matching [B,L,C] shapes");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "Wan QKV dtypes must match");
  TORCH_CHECK(q_weight.is_cuda() && k_weight.is_cuda() && q_weight.is_contiguous() && k_weight.is_contiguous(),
              "Wan norm weights must be contiguous CUDA tensors");
  TORCH_CHECK(q_weight.scalar_type() == q.scalar_type() && k_weight.scalar_type() == q.scalar_type(),
              "Wan norm weights must match the QKV dtype");
  TORCH_CHECK(freq_real.is_cuda() && freq_imag.is_cuda() && freq_real.scalar_type() == torch::kFloat32 &&
              freq_imag.scalar_type() == torch::kFloat32 && freq_real.is_contiguous() && freq_imag.is_contiguous(),
              "Wan RoPE frequency tensors must be contiguous CUDA FP32");
  const auto batch = q.size(0);
  const auto tokens = q.size(1);
  const auto channels = q.size(2);
  TORCH_CHECK(heads > 0 && channels % heads == 0, "Wan channels must divide num_heads");
  const auto head_dim = channels / heads;
  TORCH_CHECK(head_dim % 2 == 0, "Wan head_dim must be even");
  TORCH_CHECK(q_weight.numel() == channels && k_weight.numel() == channels,
              "Wan norm weights must contain C elements");
  TORCH_CHECK(freq_real.sizes() == freq_imag.sizes() && freq_real.dim() == 2 &&
              freq_real.size(0) == tokens && freq_real.size(1) == head_dim / 2,
              "Wan RoPE frequencies must have shape [L,D/2]");
  auto output_shape = std::vector<int64_t>{batch, heads, tokens, head_dim};
  auto q_out = torch::empty(output_shape, q.options());
  auto k_out = torch::empty(output_shape, k.options());
  auto v_out = torch::empty(output_shape, v.options());
  const int threads = 256;
  const int64_t rows = batch * tokens;
  const size_t shared_bytes = 2 * threads * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      q.scalar_type(), "wan_qkv_norm_rope_layout", [&] {
        wan_qkv_norm_rope_layout_kernel<scalar_t><<<
            rows, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            q_weight.data_ptr<scalar_t>(), k_weight.data_ptr<scalar_t>(),
            freq_real.data_ptr<float>(), freq_imag.data_ptr<float>(),
            q_out.data_ptr<scalar_t>(), k_out.data_ptr<scalar_t>(), v_out.data_ptr<scalar_t>(),
            rows, tokens, channels, heads, head_dim, static_cast<float>(epsilon));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_out, v_out};
}

std::vector<torch::Tensor> cosmos25_qkv_norm_rope_bshd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor rope_angles,
    int64_t heads,
    double epsilon) {
  TORCH_CHECK(
      q.is_cuda() && k.is_cuda() && v.is_cuda(),
      "Cosmos2.5 QKV inputs must be CUDA tensors");
  TORCH_CHECK(
      q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
      "Cosmos2.5 QKV inputs must be contiguous");
  TORCH_CHECK(
      q.sizes() == k.sizes() && q.sizes() == v.sizes() && q.dim() == 3,
      "Cosmos2.5 QKV inputs must have matching [B,S,C] shapes");
  TORCH_CHECK(
      q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
      "Cosmos2.5 QKV dtypes must match");
  TORCH_CHECK(
      q_weight.is_cuda() && k_weight.is_cuda() &&
          q_weight.is_contiguous() && k_weight.is_contiguous() &&
          q_weight.scalar_type() == torch::kFloat32 &&
          k_weight.scalar_type() == torch::kFloat32,
      "Cosmos2.5 norm weights must be contiguous CUDA FP32 tensors");
  TORCH_CHECK(
      rope_angles.is_cuda() && rope_angles.is_contiguous() &&
          rope_angles.scalar_type() == torch::kFloat32,
      "Cosmos2.5 RoPE angles must be contiguous CUDA FP32");
  TORCH_CHECK(
      q.get_device() == k.get_device() &&
          q.get_device() == v.get_device() &&
          q.get_device() == q_weight.get_device() &&
          q.get_device() == k_weight.get_device() &&
          q.get_device() == rope_angles.get_device(),
      "Cosmos2.5 QKV, weights, and RoPE angles must share one CUDA device");

  const auto batch = q.size(0);
  const auto tokens = q.size(1);
  const auto channels = q.size(2);
  TORCH_CHECK(
      heads > 0 && channels % heads == 0,
      "Cosmos2.5 channels must divide num_heads");
  const auto head_dim = channels / heads;
  TORCH_CHECK(
      head_dim > 0 && head_dim % 2 == 0,
      "Cosmos2.5 head_dim must be positive and even");
  TORCH_CHECK(
      q_weight.numel() == head_dim && k_weight.numel() == head_dim,
      "Cosmos2.5 norm weights must contain head_dim elements");
  TORCH_CHECK(
      rope_angles.dim() == 4 &&
          rope_angles.size(0) == tokens &&
          rope_angles.size(1) == 1 &&
          rope_angles.size(2) == 1 &&
          rope_angles.size(3) == head_dim,
      "Cosmos2.5 RoPE angles must have shape [S,1,1,D]");
  TORCH_CHECK(epsilon > 0.0, "Cosmos2.5 RMSNorm epsilon must be positive");

  auto output_shape =
      std::vector<int64_t>{batch, tokens, heads, head_dim};
  auto q_out = torch::empty(output_shape, q.options());
  auto k_out = torch::empty(output_shape, k.options());
  auto v_out = torch::empty(output_shape, v.options());
  const int threads = 256;
  const int64_t rows = batch * tokens * heads;
  const size_t shared_bytes = 2 * threads * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      q.scalar_type(), "cosmos25_qkv_norm_rope_bshd", [&] {
        cosmos25_qkv_norm_rope_bshd_kernel<scalar_t><<<
            rows, threads, shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(), q_weight.data_ptr<float>(),
            k_weight.data_ptr<float>(), rope_angles.data_ptr<float>(),
            q_out.data_ptr<scalar_t>(), k_out.data_ptr<scalar_t>(),
            v_out.data_ptr<scalar_t>(), rows, tokens, heads, head_dim,
            static_cast<float>(epsilon));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_out, v_out};
}

std::vector<torch::Tensor> role_cluster_step_q64_impl(
    torch::Tensor features,
    torch::Tensor centroids,
    int64_t max_clusters) {
  TORCH_CHECK(features.is_cuda() && centroids.is_cuda(), "Role clustering requires CUDA tensors");
  TORCH_CHECK(features.scalar_type() == torch::kFloat32 && centroids.scalar_type() == torch::kFloat32,
              "Role clustering requires FP32 features and centroids");
  TORCH_CHECK(features.is_contiguous() && centroids.is_contiguous(),
              "Role clustering inputs must be contiguous");
  TORCH_CHECK(features.dim() == 3 && centroids.dim() == 3 && features.size(0) == centroids.size(0) &&
              features.size(2) == 64 && centroids.size(2) == 64,
              "Role clustering expects features [BH,N,64], centroids [BH,C,64]");
  const auto bh = features.size(0);
  const auto tokens = features.size(1);
  const auto clusters = centroids.size(1);
  TORCH_CHECK(clusters > 0 && clusters <= max_clusters, "Unsupported role cluster count");
  auto labels = torch::empty({bh, tokens}, features.options().dtype(torch::kInt32));
  auto sizes = torch::zeros({bh, clusters}, features.options().dtype(torch::kInt32));
  auto sums = torch::zeros({bh, clusters, 64}, features.options());
  auto output = torch::empty_like(centroids);
  role_cluster_assign_accumulate_q64_kernel<<<
      bh * tokens, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      features.data_ptr<float>(), centroids.data_ptr<float>(), labels.data_ptr<int32_t>(),
      sizes.data_ptr<int32_t>(), sums.data_ptr<float>(), tokens, static_cast<int>(clusters));
  role_cluster_finalize_q64_kernel<<<
      bh * clusters, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
      sums.data_ptr<float>(), sizes.data_ptr<int32_t>(), centroids.data_ptr<float>(),
      output.data_ptr<float>(), static_cast<int>(clusters));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {labels, output, sizes};
}

std::vector<torch::Tensor> role_cluster_step_q64(
    torch::Tensor features, torch::Tensor centroids) {
  return role_cluster_step_q64_impl(features, centroids, 512);
}

std::vector<torch::Tensor> role_cluster_step_k64(
    torch::Tensor features, torch::Tensor centroids) {
  return role_cluster_step_q64_impl(features, centroids, 1024);
}

torch::Tensor svg_ear_select_budget(
    torch::Tensor scores,
    torch::Tensor cluster_sizes,
    torch::Tensor budgets) {
  TORCH_CHECK(scores.is_cuda() && cluster_sizes.is_cuda() && budgets.is_cuda(),
              "Selector inputs must be CUDA tensors");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && scores.is_contiguous() && scores.dim() == 3,
              "Selector scores must be contiguous FP32 [BH,Q,K]");
  const auto bh = scores.size(0);
  const auto q_clusters = scores.size(1);
  const auto k_clusters = scores.size(2);
  TORCH_CHECK(cluster_sizes.dim() == 2 && cluster_sizes.size(0) == bh &&
              cluster_sizes.size(1) == k_clusters,
              "Selector cluster-size shape mismatch");
  TORCH_CHECK(budgets.dim() == 2 && budgets.size(0) == bh &&
              budgets.size(1) == q_clusters,
              "Selector budget shape mismatch");
  auto sizes_i32 = cluster_sizes.to(torch::kInt32).contiguous();
  auto budgets_i32 = budgets.to(torch::kInt32).contiguous();
  auto sorted = at::sort(scores, -1, true);
  auto order = std::get<1>(sorted).contiguous();
  auto output = torch::zeros(
      {bh, q_clusters + 1, k_clusters + 1}, scores.options().dtype(torch::kBool));
  selector_budget_kernel<<<
      bh * q_clusters, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      order.data_ptr<int64_t>(), sizes_i32.data_ptr<int32_t>(), budgets_i32.data_ptr<int32_t>(),
      output.data_ptr<bool>(), nullptr, nullptr,
      static_cast<int>(q_clusters), static_cast<int>(k_clusters));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> svg_ear_select_budget_schedule(
    torch::Tensor scores,
    torch::Tensor cluster_sizes,
    torch::Tensor budgets) {
  TORCH_CHECK(scores.is_cuda() && cluster_sizes.is_cuda() && budgets.is_cuda(),
              "Selector inputs must be CUDA tensors");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && scores.is_contiguous() && scores.dim() == 3,
              "Selector scores must be contiguous FP32 [BH,Q,K]");
  const auto bh = scores.size(0);
  const auto q_clusters = scores.size(1);
  const auto k_clusters = scores.size(2);
  TORCH_CHECK(cluster_sizes.dim() == 2 && cluster_sizes.size(0) == bh &&
              cluster_sizes.size(1) == k_clusters,
              "Selector cluster-size shape mismatch");
  TORCH_CHECK(budgets.dim() == 2 && budgets.size(0) == bh &&
              budgets.size(1) == q_clusters,
              "Selector budget shape mismatch");
  auto sizes_i32 = cluster_sizes.to(torch::kInt32).contiguous();
  auto budgets_i32 = budgets.to(torch::kInt32).contiguous();
  auto sorted = at::sort(scores, -1, true);
  auto order = std::get<1>(sorted).contiguous();
  auto output = torch::zeros(
      {bh, q_clusters + 1, k_clusters + 1}, scores.options().dtype(torch::kBool));
  auto selected_ids = torch::empty(
      {bh, q_clusters, k_clusters}, scores.options().dtype(torch::kInt32));
  auto selected_counts = torch::empty(
      {bh, q_clusters}, scores.options().dtype(torch::kInt32));
  selector_budget_kernel<<<
      bh * q_clusters, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      order.data_ptr<int64_t>(), sizes_i32.data_ptr<int32_t>(), budgets_i32.data_ptr<int32_t>(),
      output.data_ptr<bool>(), selected_ids.data_ptr<int32_t>(), selected_counts.data_ptr<int32_t>(),
      static_cast<int>(q_clusters), static_cast<int>(k_clusters));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, selected_ids, selected_counts};
}

std::vector<torch::Tensor> compact_selected_clusters(
    torch::Tensor selected_mask) {
  TORCH_CHECK(
      selected_mask.is_cuda() && selected_mask.scalar_type() == torch::kBool,
      "Selected-cluster compaction requires a CUDA bool tensor");
  TORCH_CHECK(
      selected_mask.dim() >= 2,
      "Selected-cluster compaction expects [...,K]");
  auto mask = selected_mask.contiguous();
  const auto k_clusters = mask.size(-1);
  const auto rows = mask.numel() / k_clusters;
  TORCH_CHECK(
      k_clusters > 0 && k_clusters <= 4096,
      "Selected-cluster compaction supports 1..4096 K clusters");
  auto selected_ids = torch::empty(
      mask.sizes(), mask.options().dtype(torch::kInt32));
  auto count_shape = mask.sizes().vec();
  count_shape.pop_back();
  auto selected_counts = torch::empty(
      count_shape, mask.options().dtype(torch::kInt32));
  compact_selected_clusters_kernel<<<
      rows, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      mask.data_ptr<bool>(),
      selected_ids.data_ptr<int32_t>(),
      selected_counts.data_ptr<int32_t>(),
      static_cast<int>(rows),
      static_cast<int>(k_clusters));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {selected_ids, selected_counts};
}

}  // namespace

PYBIND11_MODULE(n8_kernels, module) {
  module.def(
      "wan_qkv_norm_rope_layout", &wan_qkv_norm_rope_layout,
      "Fused native-Wan Q/K RMSNorm, complex RoPE, and QKV BHLD layout");
  module.def(
      "cosmos25_qkv_norm_rope_bshd", &cosmos25_qkv_norm_rope_bshd,
      "Fused Cosmos2.5 per-head Q/K RMSNorm and TE-compatible RoPE in BSHD");
  module.def(
      "role_cluster_step_q64", &role_cluster_step_q64,
      "One exact FP32 Q-role Lloyd assignment/count/centroid step for D=64");
  module.def(
      "role_cluster_step_k64", &role_cluster_step_k64,
      "One exact FP32 K/V-role Lloyd assignment/count/centroid step for D=64");
  module.def(
      "svg_ear_select_budget", &svg_ear_select_budget,
      "Exact descending SVG-EAR selector budget and bool-mask scatter");
  module.def(
      "svg_ear_select_budget_schedule", &svg_ear_select_budget_schedule,
      "Exact selector plus ascending fixed-stride selected-cluster schedule");
  module.def(
      "compact_selected_clusters", &compact_selected_clusters,
      "Compact a CUDA bool cluster mask into ascending fixed-stride IDs and counts");
}
