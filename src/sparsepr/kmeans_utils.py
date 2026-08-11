import inspect
import math
import os
import re
import time
import warnings

import flashinfer
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from packaging.version import Version

from .flashinfer_patch import flashinfer_patch_enabled
from .timer import time_logging_decorator

_FLASHINFER_VERSION = Version(flashinfer.__version__)

# --- New functions ---


_FLASHINFER_WORKSPACE_CACHE_ENABLED = False
_FLASHINFER_KERNEL_PROFILE_ENABLED = False
_FLASHINFER_BACKEND = "auto"
_FLASHINFER_WORKSPACE_CACHE = {}
_FLASHINFER_PLAN_CACHE = {}
_FLASHINFER_PLAN_FLOAT_WORKSPACE = {}
_LAST_FLASHINFER_TIMINGS = {}
_FLASHINFER_RUNTIME_ASSERTS_ENABLED = os.getenv("SPARSEPR_FLASHINFER_RUNTIME_ASSERTS", "").lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
_KMEANS_KERNEL_PROFILE_ENABLED = os.getenv("SPARSEPR_COSMOS3_PROFILE_KMEANS_KERNEL", "").lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
_FLASH_KMEANS_USE_HEURISTIC = os.getenv("SPARSEPR_COSMOS3_FLASH_KMEANS_USE_HEURISTIC", "1").lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
_FLASH_KMEANS_INLINE_XSQ = os.getenv("SPARSEPR_COSMOS3_FLASH_KMEANS_INLINE_XSQ", "1").lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
_FLASH_KMEANS_INLINE_XSQ_ARBITRARY_D = os.getenv(
    "SPARSEPR_COSMOS3_FLASH_KMEANS_INLINE_XSQ_ARBITRARY_D", "1"
).lower() not in {"", "0", "false", "no", "off"}
_FLASH_KMEANS_INLINE_XSQ_CONFIG = os.getenv("SPARSEPR_COSMOS3_FLASH_KMEANS_INLINE_XSQ_CONFIG", "bn128_bk64_w4_s1")
_FLASH_KMEANS_INLINE_XSQ_Q_CONFIG = os.getenv(
    "SPARSEPR_COSMOS3_FLASH_KMEANS_INLINE_XSQ_Q_CONFIG", _FLASH_KMEANS_INLINE_XSQ_CONFIG
)
_FLASH_KMEANS_INLINE_XSQ_K_CONFIG = os.getenv(
    "SPARSEPR_COSMOS3_FLASH_KMEANS_INLINE_XSQ_K_CONFIG", _FLASH_KMEANS_INLINE_XSQ_CONFIG
)
_FLASH_KMEANS_INLINE_XSQ_FALLBACK_WARNED = False


def set_flash_kmeans_inline_xsq_enabled(enabled: bool) -> None:
    """Toggle the warm-start inline-x_sq path for offline A/B benchmarks."""
    global _FLASH_KMEANS_INLINE_XSQ
    _FLASH_KMEANS_INLINE_XSQ = bool(enabled)


_LAST_KMEANS_TIMINGS = {}


def set_flashinfer_workspace_cache_enabled(enabled: bool) -> None:
    global _FLASHINFER_WORKSPACE_CACHE_ENABLED
    _FLASHINFER_WORKSPACE_CACHE_ENABLED = bool(enabled)


def set_flashinfer_kernel_profile_enabled(enabled: bool) -> None:
    global _FLASHINFER_KERNEL_PROFILE_ENABLED
    _FLASHINFER_KERNEL_PROFILE_ENABLED = bool(enabled)


def set_flashinfer_backend(backend: str) -> None:
    global _FLASHINFER_BACKEND
    _FLASHINFER_BACKEND = backend


def set_kmeans_kernel_profile_enabled(enabled: bool) -> None:
    global _KMEANS_KERNEL_PROFILE_ENABLED
    _KMEANS_KERNEL_PROFILE_ENABLED = bool(enabled)


def kmeans_kernel_profile_enabled() -> bool:
    return bool(_KMEANS_KERNEL_PROFILE_ENABLED)


def clear_flashinfer_workspace_cache() -> None:
    _FLASHINFER_WORKSPACE_CACHE.clear()


def clear_flashinfer_plan_cache(*, release_wrappers: bool = False) -> None:
    """Invalidate cached sparse plans, optionally releasing their workspaces.

    A FlashInfer variable-block plan is stored on its wrapper.  Cosmos uses a
    stable wrapper per self-attention layer and invalidates the plan between
    videos. Wrappers retain plan-owned metadata but share one 128 MiB transient
    scratch buffer because layer calls are stream-ordered.
    """

    if release_wrappers:
        _FLASHINFER_PLAN_CACHE.clear()
        _FLASHINFER_PLAN_FLOAT_WORKSPACE.clear()
        return
    for entry in _FLASHINFER_PLAN_CACHE.values():
        entry["plan_token"] = None


def pop_last_flashinfer_timings() -> dict[str, float | bool | list[int] | str]:
    timings = dict(_LAST_FLASHINFER_TIMINGS)
    _LAST_FLASHINFER_TIMINGS.clear()
    return timings


def pop_last_kmeans_timings() -> dict[str, float | bool | list[int] | str]:
    timings = dict(_LAST_KMEANS_TIMINGS)
    _LAST_KMEANS_TIMINGS.clear()
    return timings


def _flashinfer_device_key(device: torch.device) -> tuple[str, int | None]:
    if device.type != "cuda":
        raise RuntimeError(f"FlashInfer sparse attention expects CUDA tensors, got device={device}.")
    return (device.type, device.index if device.index is not None else torch.cuda.current_device())


def _nvtx_push(name: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _profiled_kmeans_ms(device: torch.device, name: str, fn):
    if not _KMEANS_KERNEL_PROFILE_ENABLED or device.type != "cuda":
        return fn(), 0.0
    torch.cuda.synchronize(device)
    _nvtx_push(name)
    started = time.perf_counter()
    try:
        out = fn()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        _nvtx_pop()
    return out, elapsed_ms


def _profiled_wall_ms(device: torch.device, fn):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    out = fn()
    torch.cuda.synchronize(device)
    return out, (time.perf_counter() - started) * 1000.0


def _make_flashinfer_wrapper(
    device: torch.device,
    float_workspace_buffer: torch.Tensor | None = None,
):
    if float_workspace_buffer is None:
        float_workspace_buffer = torch.empty(128 * 1024 * 1024, device=device)
    wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(float_workspace_buffer, backend=_FLASHINFER_BACKEND)
    if _FLASHINFER_VERSION >= Version("0.6"):
        return wrapper, None
    vector_sparse_indices_buffer = torch.empty(1024 * 1024 * 1024, device=device)
    wrapper.reset_workspace_buffer(
        float_workspace_buffer=wrapper._float_workspace_buffer,
        int_workspace_buffer=wrapper._int_workspace_buffer,
        vector_sparse_indices_buffer=vector_sparse_indices_buffer,
        vector_sparse_indptr_buffer=wrapper._vector_sparse_indptr_buffer,
    )
    return wrapper, vector_sparse_indices_buffer


def _get_flashinfer_wrapper(device: torch.device):
    if not _FLASHINFER_WORKSPACE_CACHE_ENABLED:
        return _make_flashinfer_wrapper(device), False, False
    key = _flashinfer_device_key(device)
    cache_hit = key in _FLASHINFER_WORKSPACE_CACHE
    if key not in _FLASHINFER_WORKSPACE_CACHE:
        _FLASHINFER_WORKSPACE_CACHE[key] = _make_flashinfer_wrapper(device)
    return _FLASHINFER_WORKSPACE_CACHE[key], True, cache_hit


def _get_flashinfer_plan_wrapper(device: torch.device, plan_cache_key):
    """Return one persistent wrapper for a caller-owned sparse topology."""

    key = (_flashinfer_device_key(device), plan_cache_key)
    cache_hit = key in _FLASHINFER_PLAN_CACHE
    if not cache_hit:
        device_key = _flashinfer_device_key(device)
        if device_key not in _FLASHINFER_PLAN_FLOAT_WORKSPACE:
            _FLASHINFER_PLAN_FLOAT_WORKSPACE[device_key] = torch.empty(
                128 * 1024 * 1024, device=device
            )
        _FLASHINFER_PLAN_CACHE[key] = {
            # All layer calls are ordered on the same CUDA stream, so their
            # transient split-K scratch can safely share one device buffer.
            # Plan-owned indptr/indices and plan_info remain wrapper-local.
            "wrapper_pair": _make_flashinfer_wrapper(
                device, _FLASHINFER_PLAN_FLOAT_WORKSPACE[device_key]
            ),
            "plan_token": None,
        }
    return _FLASHINFER_PLAN_CACHE[key], cache_hit


_FLASHINFER_LSE_NATURAL_SCALE: dict[tuple[str, int | None], float] = {}


def _flashinfer_lse_natural_scale(device: torch.device) -> float:
    """Return the factor that converts FlashInfer's LSE output to natural log.

    FlashInfer kernels historically store LSE in log2 units; some versions and
    backends return natural log.  Rather than hardcoding a convention that can
    change between releases, run one tiny dense block-sparse problem and
    compare against torch.logsumexp.  Cached per device.
    """
    key = _flashinfer_device_key(device)
    cached = _FLASHINFER_LSE_NATURAL_SCALE.get(key)
    if cached is not None:
        return cached

    tokens, dim = 64, 64
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn((1, tokens, dim), generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
    k = torch.randn((1, tokens, dim), generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
    v = torch.randn((1, tokens, dim), generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)

    (wrapper, _), _, _ = _get_flashinfer_wrapper(device)
    with flashinfer_patch_enabled():
        wrapper.plan(
            block_mask_map=torch.ones((1, 1, 1), dtype=torch.bool, device=device),
            block_row_sz=torch.full((1, 1), tokens, dtype=torch.int32, device=device),
            block_col_sz=torch.full((1, 1), tokens, dtype=torch.int32, device=device),
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim=dim,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )
    _, lse = wrapper.run(q, k, v, return_lse=True)
    lse = lse.reshape(-1).float()

    logits = (q.float() @ k.float().transpose(1, 2)) * (dim**-0.5)
    lse_ref = torch.logsumexp(logits, dim=-1).reshape(-1)

    natural_err = (lse - lse_ref).abs().max().item()
    log2_err = (lse * math.log(2.0) - lse_ref).abs().max().item()
    if natural_err < 0.05:
        scale = 1.0
    elif log2_err < 0.05:
        scale = math.log(2.0)
    else:
        raise RuntimeError(
            "Could not determine FlashInfer LSE convention: "
            f"natural_err={natural_err:.4f}, log2_err={log2_err:.4f}. "
            f"flashinfer=={flashinfer.__version__}."
        )
    _FLASHINFER_LSE_NATURAL_SCALE[key] = scale
    print(f"[FI-LSE] flashinfer LSE convention: {'natural' if scale == 1.0 else 'log2'} (scale={scale:.6f})")
    return scale


@time_logging_decorator("Level 4- density calculation")
def density_calculation(dynamic_map, q_cluster_sizes, k_cluster_sizes):
    """
    Calculate the density of the dynamic map. Currently only batch size = 1 and head size = 1 are supported.

    Input:
        dynamic_map: [cfg, num_heads, qc_num, kc_num]
        q_cluster_sizes: [cfg, num_heads, qc_num]
        k_cluster_sizes: [cfg, num_heads, kc_num]
    """
    cfg, num_heads, qc_num, kc_num = dynamic_map.shape

    # Calculate the block size of each block
    clustered_block_size = q_cluster_sizes[:, :, :, None] * k_cluster_sizes[:, :, None, :]
    masked_block_size = clustered_block_size * dynamic_map

    # Calculate the density of each block
    density = torch.sum(masked_block_size, dim=(2, 3)) / torch.sum(clustered_block_size, dim=(2, 3))
    return density


# --- Functions from analyze/kmeans_rapidai.py ---


def pairwise_distance(x, y):
    """
    Computes pairwise squared Euclidean distance between two sets of points.
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = torch.clamp(x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1)), min=0.0)
    return dist


def kmeans_predict(centroids, input_tensor):  # Removed unused params argument
    """
    Predict the labels for the input tensor using the centroids.
    """
    input_tensor = input_tensor.to(torch.float32)
    dist = pairwise_distance(input_tensor, centroids)
    labels = torch.argmin(dist, dim=1)
    return labels


def kmeans_rapidai(
    tensor, k, max_iter=5, tol=1e-4, init_method="Array", centroids_init=None
):  # Renamed centroids to centroids_init
    """
    Performs K-means clustering using cuVS.
    """
    from cuvs.cluster.kmeans import KMeansParams, fit

    assert tensor.dtype == torch.float32, "Tensor must be float32 for cuVS KMeans"
    assert tensor.ndim == 2, f"Tensor must be 2D, but got {tensor.shape}"
    # assert init_method == "Array", "init_method must be 'Array' for now"

    L, D = tensor.shape

    # cuVS KMeans in RAPIDS >=23.10 uses 'centroids_init' for initial centroids
    current_centroids = centroids_init
    if current_centroids is None:
        # Default init: cuVS handles KMeansPlusPlus if centroids_init is None and init_method is KMeansPlusPlus
        # If you need to pass an empty tensor for cuVS to initialize:
        current_centroids = torch.empty(k, D, device=tensor.device, dtype=torch.float32)  # Or pass None
    else:
        assert current_centroids.dtype == torch.float32, "Initial centroids must be float32"
        assert current_centroids.shape == (
            k,
            D,
        ), f"Initial centroids shape mismatch, got {current_centroids.shape}, expected ({k}, {D})"
        # cuVS uses 'init_method="Array"' when 'centroids_init' is provided.

    # import IPython; IPython.embed()

    params = KMeansParams(
        n_clusters=k, max_iter=max_iter, tol=tol, init_method=init_method
    )  # Changed init_method to init

    # Call fit with centroids_init (can be None)
    new_centroids, inertia, n_iter_ = fit(params, tensor, current_centroids)  # Added handle=None

    labels = kmeans_predict(new_centroids, tensor)
    return labels, new_centroids, n_iter_


@triton.jit
def _centroid_update_kernel(
    x_ptr,  # *f16  [B, N, D]
    cluster_ptr,  # *i32  [B, N]
    sum_ptr,  # *f32  [B, K, D]
    count_ptr,  # *i32  [B, K]
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,  # number of dims processed per program
):
    """Each program processes 1 point (token) across BLOCK_D dimensions with atomics."""
    pid = tl.program_id(axis=0)
    token_idx = pid  # range: [0, B * N)

    # Derive (b, n) indices
    b = token_idx // N
    n = token_idx % N

    # Pointers to the token features and its cluster id
    x_offset = (b * N + n) * D
    x_ptr = x_ptr + x_offset

    cluster_idx = tl.load(cluster_ptr + b * N + n)  # int32

    # Guard for invalid cluster ids (should not happen)
    cluster_idx = tl.where(cluster_idx < K, cluster_idx, 0)

    # Base pointer for this centroid in the output sum tensor
    centroid_base = (b * K + cluster_idx) * D

    # Process feature vector in chunks of BLOCK_D
    offs = tl.arange(0, BLOCK_D)
    for d_start in range(0, D, BLOCK_D):
        mask = offs + d_start < D
        feats = tl.load(x_ptr + d_start + offs, mask=mask, other=0.0)
        feats = feats.to(tl.float32)

        dest_ptr = sum_ptr + centroid_base + d_start + offs
        tl.atomic_add(dest_ptr, feats, mask=mask)

    # Update counts (only once per point)
    tl.atomic_add(count_ptr + b * K + cluster_idx, 1)


def triton_centroid_update_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
    """Compute centroids using custom Triton kernel.

    Args:
        x_norm (Tensor): (B, N, D) normalized input vectors (float16/float32)
        cluster_ids (LongTensor): (B, N) cluster assignment per point
        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x_norm)

    Returns:
        Tensor: (B, K, D) updated and L2-normalized centroids (dtype == x_norm.dtype)
    """
    assert x_norm.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # Allocate accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x_norm.device, dtype=torch.float32)
    centroid_counts = torch.zeros((B, K), device=x_norm.device, dtype=torch.int32)

    # Launch Triton kernel – one program per token
    total_tokens = B * N
    BLOCK_D = 128  # tuneable

    grid = (total_tokens,)
    _centroid_update_kernel[grid](
        x_norm,
        cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_counts,
        B,
        N,
        D,
        K,
        BLOCK_D=BLOCK_D,
    )

    # Compute means; keep old centroid if empty cluster
    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f

    # For clusters with zero count, revert to old centroids
    zero_mask = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

    centroids = centroids.to(x_norm.dtype)
    centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids


def torch_loop_centroid_update_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
    """Reference Python implementation (double for-loop)"""
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    new_centroids = torch.zeros_like(old_centroids)
    for b in range(B):
        for k in range(K):
            mask = cluster_ids[b] == k
            if mask.any():
                new_centroids[b, k] = F.normalize(x_norm[b][mask].mean(dim=0, dtype=x_norm.dtype), p=2, dim=0)
            else:
                new_centroids[b, k] = old_centroids[b, k]
    return new_centroids


def triton_centroid_update_euclid(x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
    """Compute centroids for Euclidean KMeans using Triton.

    Args:
        x (Tensor): (B, N, D) input vectors (float16/float32)
        cluster_ids (LongTensor): (B, N) cluster assignment per point
        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x)

    Returns:
        Tensor: (B, K, D) updated centroids (dtype == x.dtype)
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # Allocate accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_counts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    total_tokens = B * N
    BLOCK_D = 128  # tuneable
    grid = (total_tokens,)

    _centroid_update_kernel[grid](
        x,
        cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_counts,
        B,
        N,
        D,
        K,
        BLOCK_D=BLOCK_D,
    )

    # Compute means; keep old centroid if empty cluster
    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f

    # For clusters with zero count, revert to old centroids
    zero_mask = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

    return centroids.to(x.dtype)


# ------------------------------ NEW: chunk-wise centroid update (sorted ids) ------------------------------


@triton.jit
def _centroid_update_chunk_kernel(
    x_ptr,  # *f16 / *f32 [B, N, D] – ORIGINAL ORDER
    sorted_idx_ptr,  # *i32        [B, N]    – indices after sort
    sorted_cluster_ptr,  # *i32        [B, N]    – cluster ids in sorted order
    sum_ptr,  # *f32        [B, K, D]
    count_ptr,  # *i32        [B, K]
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,  # how many tokens (points) each program processes
    BLOCK_D: tl.constexpr,
):
    """Each program processes **BLOCK_N consecutive, already-sorted tokens**.

    Because the tokens are sorted by cluster id, identical ids appear in
    contiguous runs.  We therefore accumulate a local sum/count for the
    current run and perform **a single atomic update per run**, instead of
    per-token.
    """
    # program indices – 2-D launch grid: (chunk_id, batch_id)
    pid_chunk = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    b = pid_b
    chunk_start = pid_chunk * BLOCK_N  # position of the first token handled by this program

    # Nothing to do – out of range
    if chunk_start >= N:
        return

    # base pointers for this batch
    idx_batch_base = sorted_idx_ptr + b * N
    cid_batch_base = sorted_cluster_ptr + b * N
    x_batch_base = x_ptr + b * N * D  # for pointer arithmetic

    # helper aranges
    offs_token = tl.arange(0, BLOCK_N)
    offs_dim = tl.arange(0, BLOCK_D)
    valid_dim = offs_dim < D

    # first token index & validity mask
    token_idx = chunk_start + offs_token
    valid_tok = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    # Load first cluster id to initialise the running accumulator
    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(cid_batch_base + token_idx, mask=valid_tok, other=-1)

    all_tokens_idxs = tl.load(idx_batch_base + token_idx, mask=valid_tok, other=-1)  # [BLOCK_N]

    load_mask = all_tokens_idxs[:, None] * D + offs_dim[None, :]

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            cluster_feats = tl.load(
                x_batch_base + load_mask,
                mask=cluster_mask[:, None] & valid_dim[None, :],
                other=0.0,
            )  # [BLOCK_N, BLOCK_D]
            cluster_feats = cluster_feats.to(tl.float32)
            sum_feats = tl.sum(cluster_feats, axis=0)
            dest_ptr = sum_ptr + (b * K + cid) * D + offs_dim
            tl.atomic_add(dest_ptr, sum_feats, mask=valid_dim)
            tl.atomic_add(count_ptr + b * K + cid, cluster_size)


# ---------------------------------------------------------------------------------------------


def triton_centroid_update_sorted_cosine(
    x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor, *, BLOCK_N: int = 256
):
    """Fast centroid update assuming **cluster_ids are sorted along N**.

    This helper will sort the assignments (together with `x_norm`) and launch the
    chunk kernel above.  Compared to the naive per-token kernel it performs *one
    atomic add per run of identical ids* instead of per token, providing large
    speed-ups when clusters are reasonably sized.
    """
    assert x_norm.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA"
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # -------- sort per-batch --------
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    # accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x_norm.device, dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K), device=x_norm.device, dtype=torch.int32)

    grid = (triton.cdiv(N, BLOCK_N), B)
    block_d = max(16, 1 << (D - 1).bit_length())
    _centroid_update_chunk_kernel[grid](
        x_norm,
        sorted_idx_int,
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        B,
        N,
        D,
        K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
    )

    # finalise – convert to means, handle empty clusters, renormalise
    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    empty_mask = (centroid_cnts == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
    centroids = centroids.to(x_norm.dtype)
    centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids


def triton_centroid_update_sorted_euclid(
    x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor, *, BLOCK_N: int = 256
):
    """Fast centroid update for *Euclidean* KMeans assuming cluster IDs are pre-sorted.

    Parameters
    ----------
    x : Tensor [B, N, D]
        Input feature vectors (no normalization assumed).
    cluster_ids : LongTensor [B, N]
        Cluster assignment for each point.
    old_centroids : Tensor [B, K, D]
        Previous centroids (used to fill empty clusters).
    BLOCK_N : int, optional
        Tokens per Triton program (affects occupancy/perf).
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]

    # Batch-wise sort of cluster assignments
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    grid = (triton.cdiv(N, BLOCK_N), B)
    block_d = max(16, 1 << (D - 1).bit_length())
    _centroid_update_chunk_kernel[grid](
        x,  # original features
        sorted_idx_int,  # gather indices
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        B,
        N,
        D,
        K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
    )

    # Convert sums to means; replace empty clusters with old centroids
    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    empty_mask = (centroid_cnts == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
    return centroids.to(x.dtype), centroid_cnts


# ===============================================================
# Triton kernel: compute nearest-centroid IDs (Euclidean distance)
# Inputs:
#   x           : (B, N, D)  float16 / float32
#   centroids   : (B, K, D)  same dtype as x
#   x_sq        : (B, N)     float32 – pre-computed ||x||^2 per point
# Output:
#   cluster_ids : (B, N)     int32   – nearest centroid index per point
# ===============================================================


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# -----------------------------------------------------------------------------
# Auto-tuning setup – explore various tile sizes / warp counts
# -----------------------------------------------------------------------------

_TUNE_CONFIGS = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=4, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [32, 64, 128]
    for wp in [4, 8]
]


def _cfg_keep(conf):
    """Basic heuristic to prune unbalanced configs."""
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    # Avoid tiny tiles on many warps
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    return True


_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))


def _parse_inline_xsq_config(value: str) -> dict[str, int]:
    match = re.fullmatch(r"bn(\d+)_bk(\d+)_w(\d+)_s(\d+)", value.strip())
    if not match:
        raise RuntimeError(f"Invalid SPARSEPR_COSMOS3_FLASH_KMEANS_INLINE_XSQ_CONFIG={value!r}; expected bn128_bk64_w4_s1.")
    block_n, block_k, num_warps, num_stages = (int(part) for part in match.groups())
    for name, tile in (("BLOCK_N", block_n), ("BLOCK_K", block_k)):
        if tile < 16 or tile & (tile - 1):
            raise RuntimeError(f"Invalid inline x_sq {name}={tile}; Triton tile sizes must be powers of two >= 16.")
    if num_warps not in {1, 2, 4, 8}:
        raise RuntimeError(f"Invalid inline x_sq num_warps={num_warps}; expected one of 1, 2, 4, 8.")
    if num_stages < 1:
        raise RuntimeError(f"Invalid inline x_sq num_stages={num_stages}; expected >= 1.")
    return {
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _inline_xsq_config_for_clusters(n_clusters: int) -> tuple[str, dict[str, int]]:
    # Cosmos3 uses 300 Q centroids and 1000 K centroids by default. Allow the
    # larger-K stream to be tuned separately without threading stream names
    # through the generic k-means API.
    config_name = _FLASH_KMEANS_INLINE_XSQ_Q_CONFIG if n_clusters <= 512 else _FLASH_KMEANS_INLINE_XSQ_K_CONFIG
    return config_name, _parse_inline_xsq_config(config_name)


@triton.jit
def _euclid_assign_inline_xsq_kernel(
    x_ptr,
    c_ptr,
    c_sq_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    # ``tl.arange`` and the K dimension consumed by ``tl.dot`` must use a
    # power-of-two extent.  Pad arbitrary feature widths (Cosmos3 has a D=48
    # role stream) to BLOCK_D and mask the tail to zero.
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    x_ptrs = x_ptr + pid_b * stride_x_b + n_offsets[:, None] * stride_x_n + offs_d[None, :] * stride_x_d
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    x_tile_f = x_tile.to(tl.float32)
    x_sq_tile = tl.sum(x_tile_f * x_tile_f, axis=1)

    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        c_ptrs = c_ptr + pid_b * stride_c_b + k_offsets[None, :] * stride_c_k + offs_d[:, None] * stride_c_d
        c_tile = tl.load(c_ptrs, mask=d_mask[:, None] & k_mask[None, :], other=0.0)
        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        cross = tl.dot(x_tile, c_tile).to(tl.float32)
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)
        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


def euclid_assign_inline_xsq_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    c_sq: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    config: dict[str, int] | None = None,
) -> torch.Tensor:
    assert x.is_cuda and centroids.is_cuda and c_sq.is_cuda, "All tensors must be on CUDA"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"
    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"
    assert c_sq.shape == (B, K), "c_sq shape mismatch"
    if D & (D - 1) and not _FLASH_KMEANS_INLINE_XSQ_ARBITRARY_D:
        raise RuntimeError(f"Inline x_sq assign currently requires power-of-two D, got D={D}.")
    block_d = max(16, 1 << (D - 1).bit_length())
    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
    cfg = config or _parse_inline_xsq_config(_FLASH_KMEANS_INLINE_XSQ_CONFIG)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()
    grid = (triton.cdiv(N, cfg["BLOCK_N"]), B)
    _euclid_assign_inline_xsq_kernel[grid](
        x,
        centroids,
        c_sq,
        out,
        B,
        N,
        K,
        D,
        block_d,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_c_b,
        stride_c_k,
        stride_c_d,
        stride_csq_b,
        stride_csq_k,
        stride_out_b,
        stride_out_n,
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


@triton.autotune(_TUNE_CONFIGS, key=["N", "K"])
@triton.jit
def _euclid_assign_kernel(
    x_ptr,  # *f16 / *f32 [B, N, D]
    c_ptr,  # *f16 / *f32 [B, K, D]
    x_sq_ptr,  # *f32         [B, N]
    out_ptr,  # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Each program handles a tile of BLOCK_N points for a given batch element.

    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
    maintains the running minimum distance as well as the corresponding index
    for every point in the tile.
    """
    pid_n = tl.program_id(0)  # tile index along N dimension
    pid_b = tl.program_id(1)  # batch index

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load x tile  (BLOCK_N, D)
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D)
    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
    x_ptrs = x_ptr + pid_b * stride_x_b + n_offsets[:, None] * stride_x_n + offs_d[None, :] * stride_x_d
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_tile = x_tile  # compute in f32

    # Pre-load x_sq for the tile  (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)  # large number
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in chunks of BLOCK_K
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load centroid tile  (D, BLOCK_K)
        c_ptrs = c_ptr + pid_b * stride_c_b + k_offsets[None, :] * stride_c_k + offs_d[:, None] * stride_c_d
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
        c_tile = c_tile

        # Compute centroid squared norms (BLOCK_K,)
        cent_sq = tl.sum(c_tile * c_tile, axis=0).to(tl.float32)

        # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # float32

        # Squared Euclidean distance
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)

        # Mask out invalid centroid columns before reduction
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


# ---------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------


def euclid_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    out: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    """Return nearest-centroid indices using Triton kernel.

    Args:
        x         : (B, N, D) float16 / float32 (on CUDA)
        centroids : (B, K, D) same dtype/device as x
        x_sq      : (B, N)    float32 – ||x||^2 per point (on CUDA)

    Returns:
        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda, "All tensors must be on CUDA"
    # assert x.dtype in (torch.float16, torch.float32), "x must be fp16/fp32"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"
    assert x_sq.shape == (B, N), "x_sq shape mismatch"

    # x = x.contiguous()
    # centroids = centroids.contiguous()
    # x_sq = x_sq.contiguous()

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int64)

    # Strides (in elements)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    _euclid_assign_kernel[grid](
        x,
        centroids,
        x_sq,
        out,
        B,
        N,
        K,
        D,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_c_b,
        stride_c_k,
        stride_c_d,
        stride_xsq_b,
        stride_xsq_n,
        stride_out_b,
        stride_out_n,
    )
    return out


# 1. Euclidean
@time_logging_decorator("Level 5 - batch kmeans euclid iter")
def _euclid_iter(x, x_sq, centroids):
    # cent_sq = (centroids ** 2).sum(dim=-1)
    # cross = torch.einsum('bnd,bkd->bnk', x, centroids)
    # dist_sq = (x_sq[:,:,None] + cent_sq[:,None,:] - 2.0 * cross).clamp_min_(0.0)

    cluster_ids = euclid_assign_triton(x, centroids, x_sq)
    centroids_new, cluster_sizes = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)

    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids, cluster_sizes


# 2. Cosine
def _cosine_iter(x_norm, centroids):
    cos_sim = torch.einsum("bnd,bkd->bnk", x_norm, centroids)
    cluster_ids = cos_sim.argmax(dim=-1)
    centroids_new = triton_centroid_update_cosine(x_norm, cluster_ids, centroids)
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids


# 3. Dot-product
def _dot_iter(x, centroids):
    sim = torch.einsum("bnd,bkd->bnk", x, centroids)
    cluster_ids = sim.argmax(dim=-1)
    centroids_new = triton_centroid_update_cosine(x, cluster_ids, centroids)
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids


COMPILE_FLAG = False

# Try to compile; if PyTorch < 2.0 or compile is not available, fallback to original function
try:
    if COMPILE_FLAG:
        _euclid_iter_compiled = torch.compile(_euclid_iter, dynamic=True, mode="reduce-overhead")
        _cosine_iter_compiled = torch.compile(_cosine_iter, dynamic=True, mode="reduce-overhead")
        _dot_iter_compiled = torch.compile(_dot_iter, dynamic=True, mode="reduce-overhead")
    else:
        _euclid_iter_compiled = _euclid_iter
        _cosine_iter_compiled = _cosine_iter
        _dot_iter_compiled = _dot_iter
except Exception:  # pragma: no cover
    _euclid_iter_compiled = _euclid_iter
    _cosine_iter_compiled = _cosine_iter
    _dot_iter_compiled = _dot_iter


def _cluster_sizes_from_labels(
    cluster_ids: torch.Tensor,
    n_clusters: int,
    *,
    profile_prefix: str = "cluster_sizes",
    profile_timings: dict[str, float] | None = None,
) -> torch.Tensor:
    B, N = cluster_ids.shape
    if not _KMEANS_KERNEL_PROFILE_ENABLED or profile_timings is None:
        ones = torch.ones((B, N), device=cluster_ids.device, dtype=torch.int64)
        cluster_sizes = torch.zeros((B, n_clusters), device=cluster_ids.device, dtype=torch.int64)
        return cluster_sizes.scatter_add_(1, cluster_ids.long(), ones)

    device = cluster_ids.device
    ones, ms = _profiled_kmeans_ms(
        device,
        f"{profile_prefix}.ones_alloc",
        lambda: torch.ones((B, N), device=device, dtype=torch.int64),
    )
    profile_timings[f"{profile_prefix}_ones_alloc_ms"] = ms

    cluster_sizes, ms = _profiled_kmeans_ms(
        device,
        f"{profile_prefix}.sizes_alloc",
        lambda: torch.zeros((B, n_clusters), device=device, dtype=torch.int64),
    )
    profile_timings[f"{profile_prefix}_sizes_alloc_ms"] = ms

    labels_long, ms = _profiled_kmeans_ms(
        device,
        f"{profile_prefix}.labels_cast",
        lambda: cluster_ids.long(),
    )
    profile_timings[f"{profile_prefix}_labels_cast_ms"] = ms

    cluster_sizes, ms = _profiled_kmeans_ms(
        device,
        f"{profile_prefix}.scatter_add",
        lambda: cluster_sizes.scatter_add_(1, labels_long, ones),
    )
    profile_timings[f"{profile_prefix}_scatter_add_ms"] = ms
    profile_timings[f"{profile_prefix}_ms"] = (
        profile_timings[f"{profile_prefix}_ones_alloc_ms"]
        + profile_timings[f"{profile_prefix}_sizes_alloc_ms"]
        + profile_timings[f"{profile_prefix}_labels_cast_ms"]
        + profile_timings[f"{profile_prefix}_scatter_add_ms"]
    )
    return cluster_sizes


def batch_kmeans_Euclid_subsampled(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int,
    *,
    sample_tokens: int,
    sample_seed: int,
    init_centroids: torch.Tensor | None = None,
    tol: float = 1e-4,
    verbose: bool = False,
    backend: str = "cuvs",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Fit k-means on a token subsample, then assign and recompute on all tokens."""
    if max_iters < 0:
        raise RuntimeError(f"max_iters must be non-negative, got {max_iters}.")
    _, n_tokens, _ = x.shape
    if sample_tokens <= 0 or sample_tokens >= n_tokens or max_iters == 0:
        return batch_kmeans_Euclid(
            x,
            n_clusters,
            max_iters=max_iters,
            tol=tol,
            init_centroids=init_centroids,
            verbose=verbose,
            backend=backend,
        )

    sample_n = max(n_clusters, min(sample_tokens, n_tokens))
    generator = torch.Generator(device="cpu").manual_seed(sample_seed + n_tokens * 37 + n_clusters * 101)
    sample_idx = torch.randperm(n_tokens, generator=generator)[:sample_n].to(x.device)
    x_sample = x[:, sample_idx, :].contiguous()

    _, fit_centroids, _, n_iters = batch_kmeans_Euclid(
        x_sample,
        n_clusters,
        max_iters=max_iters,
        tol=tol,
        init_centroids=init_centroids,
        verbose=verbose,
        backend=backend,
    )
    if not x.is_cuda:
        raise RuntimeError("Subsampled k-means assign-all requires a CUDA input tensor.")
    cluster_ids, centroids, cluster_sizes, _ = _euclid_assign_only(
        x,
        fit_centroids,
        n_clusters,
        recompute_centroids=True,
    )
    return cluster_ids, centroids, cluster_sizes, int(n_iters)


def _euclid_assign_only(
    x: torch.Tensor,
    centroids: torch.Tensor,
    n_clusters: int,
    *,
    recompute_centroids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Assign all points to fixed centroids using the CUDA Triton kernel only."""
    if centroids is None:
        raise RuntimeError("Euclidean assign-only mode requires init_centroids.")
    if not x.is_cuda:
        raise RuntimeError(
            "Euclidean assign-only k-means requires a CUDA tensor "
            f"(device={x.device}). The chunked Python assign path was removed."
        )
    timings: dict[str, float | bool | list[int] | str] = {}
    B, N, D = x.shape
    if _KMEANS_KERNEL_PROFILE_ENABLED:
        _LAST_KMEANS_TIMINGS.clear()
        timings.update(
            {
                "kmeans_backend_path": "assign_only",
                "kmeans_recompute_centroids": bool(recompute_centroids),
                "kmeans_shape": [int(B), int(N), int(D)],
                "kmeans_clusters": int(n_clusters),
            }
        )
    x, x_ms = _profiled_kmeans_ms(x.device, "kmeans.assign_only.input_contiguous", lambda: x.contiguous())
    centroids, c_ms = _profiled_kmeans_ms(
        x.device,
        "kmeans.assign_only.centroids_contiguous",
        lambda: centroids.contiguous().view(B, n_clusters, D),
    )
    if _KMEANS_KERNEL_PROFILE_ENABLED:
        timings["input_contiguous_ms"] = x_ms + c_ms
    x_sq, ms = _profiled_kmeans_ms(
        x.device,
        "kmeans.assign_only.x_sq",
        lambda: (x.float() ** 2).sum(dim=-1),
    )
    if _KMEANS_KERNEL_PROFILE_ENABLED:
        timings["x_sq_ms"] = ms
    cluster_ids, ms = _profiled_kmeans_ms(
        x.device,
        "kmeans.assign_only.assign_triton",
        lambda: euclid_assign_triton(x, centroids, x_sq).long(),
    )
    if _KMEANS_KERNEL_PROFILE_ENABLED:
        timings["assign_triton_ms"] = ms
    if recompute_centroids:
        (centroids, cluster_sizes), ms = _profiled_kmeans_ms(
            x.device,
            "kmeans.assign_only.centroid_update",
            lambda: triton_centroid_update_sorted_euclid(x, cluster_ids, centroids),
        )
        if _KMEANS_KERNEL_PROFILE_ENABLED:
            timings["centroid_update_ms"] = ms
    else:
        cluster_sizes = _cluster_sizes_from_labels(
            cluster_ids,
            n_clusters,
            profile_prefix="cluster_sizes",
            profile_timings=timings,
        )
    if _KMEANS_KERNEL_PROFILE_ENABLED:
        _LAST_KMEANS_TIMINGS.update(timings)
    return cluster_ids, centroids, cluster_sizes, 0


def _batch_kmeans_Euclid_flash_inline_xsq(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int,
    tol: float,
    init_centroids: torch.Tensor,
    *,
    verbose: bool = False,
    profile_timings: dict[str, float | bool | list[int] | str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Warm-start Flash-KMeans with ||x||^2 fused into arbitrary-D assign."""
    B, N, D = x.shape
    if init_centroids is None:
        raise RuntimeError("Inline x_sq Flash-KMeans path requires warm-start centroids.")
    if D & (D - 1) and not _FLASH_KMEANS_INLINE_XSQ_ARBITRARY_D:
        raise RuntimeError(f"Inline x_sq Flash-KMeans path requires power-of-two D, got D={D}.")
    device = x.device
    cfg_name, cfg = _inline_xsq_config_for_clusters(n_clusters)
    if profile_timings is not None:
        profile_timings["kmeans_backend_path"] = "flash_inline_xsq"
        profile_timings["inline_xsq_config"] = str(cfg_name)

    if _KMEANS_KERNEL_PROFILE_ENABLED and device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    centroids = init_centroids.view(B, n_clusters, D)
    cluster_ids = torch.empty((B, N), device=device, dtype=torch.int32)
    c_sq = torch.empty((B, n_clusters), device=device, dtype=torch.float32)
    cluster_sizes = None
    it = -1
    for it in range(max_iters):
        _, ms = _profiled_kmeans_ms(
            device,
            "kmeans.flash_inline_xsq.c_sq",
            lambda: c_sq.copy_((centroids.to(torch.float32) ** 2).sum(-1)),
        )
        if profile_timings is not None:
            profile_timings["inline_xsq_c_sq_ms"] = profile_timings.get("inline_xsq_c_sq_ms", 0.0) + ms

        cluster_ids, ms = _profiled_kmeans_ms(
            device,
            "kmeans.flash_inline_xsq.assign",
            lambda: euclid_assign_inline_xsq_triton(x, centroids, c_sq, cluster_ids, config=cfg),
        )
        if profile_timings is not None:
            profile_timings["inline_xsq_assign_ms"] = profile_timings.get("inline_xsq_assign_ms", 0.0) + ms

        (centroids_new, cluster_sizes), ms = _profiled_kmeans_ms(
            device,
            "kmeans.flash_inline_xsq.centroid_update",
            lambda: triton_centroid_update_sorted_euclid(x, cluster_ids, centroids),
        )
        if profile_timings is not None:
            profile_timings["inline_xsq_centroid_update_ms"] = (
                profile_timings.get("inline_xsq_centroid_update_ms", 0.0) + ms
            )

        center_shift, ms = _profiled_kmeans_ms(
            device,
            "kmeans.flash_inline_xsq.shift",
            lambda: (centroids_new - centroids).norm(dim=-1).max(),
        )
        if profile_timings is not None:
            profile_timings["inline_xsq_shift_ms"] = profile_timings.get("inline_xsq_shift_ms", 0.0) + ms

        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break

        centroids, ms = _profiled_kmeans_ms(
            device,
            "kmeans.flash_inline_xsq.clone",
            lambda: centroids_new.clone(),
        )
        if profile_timings is not None:
            profile_timings["inline_xsq_clone_ms"] = profile_timings.get("inline_xsq_clone_ms", 0.0) + ms

    if _KMEANS_KERNEL_PROFILE_ENABLED and device.type == "cuda":
        torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - started) * 1000.0
    if profile_timings is not None:
        profile_timings["flash_kmeans_ms"] = total_ms
        profile_timings["inline_xsq_total_ms"] = total_ms
        profile_timings["kmeans_iters_actual"] = int(it + 1)
    if cluster_sizes is None:
        raise RuntimeError("Inline x_sq Flash-KMeans path requires max_iters >= 1.")
    return cluster_ids, centroids, cluster_sizes.long(), int(it + 1)


def _batch_kmeans_Euclid_flash(x, n_clusters, max_iters=100, tol=1e-4, init_centroids=None, verbose=False):
    if max_iters == 0:
        return _euclid_assign_only(x, init_centroids, n_clusters, recompute_centroids=False)
    if not x.is_cuda:
        raise RuntimeError("Flash-KMeans backend requires a CUDA input tensor.")
    if x.ndim != 3:
        raise RuntimeError(f"Flash-KMeans expects [B, N, D] input, got shape {tuple(x.shape)}.")
    timings: dict[str, float | bool | list[int] | str] = {}
    if _KMEANS_KERNEL_PROFILE_ENABLED:
        _LAST_KMEANS_TIMINGS.clear()
        B, N, D = x.shape
        timings.update(
            {
                "kmeans_backend_path": "flash",
                "kmeans_shape": [int(B), int(N), int(D)],
                "kmeans_clusters": int(n_clusters),
                "kmeans_max_iters": int(max_iters),
                "kmeans_has_init_centroids": bool(init_centroids is not None),
                "flash_kmeans_use_heuristic": bool(_FLASH_KMEANS_USE_HEURISTIC),
                "flash_kmeans_inline_xsq": bool(_FLASH_KMEANS_INLINE_XSQ),
            }
        )
    x, x_ms = _profiled_kmeans_ms(x.device, "kmeans.flash.input_contiguous", lambda: x.contiguous())
    c_ms = 0.0
    if init_centroids is not None:
        init_centroids, c_ms = _profiled_kmeans_ms(
            x.device,
            "kmeans.flash.init_centroids_contiguous",
            lambda: init_centroids.contiguous(),
        )
    if _KMEANS_KERNEL_PROFILE_ENABLED:
        timings["input_contiguous_ms"] = x_ms + c_ms
    if _FLASH_KMEANS_INLINE_XSQ and init_centroids is not None:
        try:
            out = _batch_kmeans_Euclid_flash_inline_xsq(
                x,
                n_clusters,
                max_iters,
                tol,
                init_centroids,
                verbose=verbose,
                profile_timings=timings,
            )
        except Exception as exc:
            global _FLASH_KMEANS_INLINE_XSQ_FALLBACK_WARNED
            if not _FLASH_KMEANS_INLINE_XSQ_FALLBACK_WARNED:
                warnings.warn(
                    "SPARSEPR_COSMOS3_FLASH_KMEANS_INLINE_XSQ=1 fell back to standard Flash-KMeans after an inline "
                    f"x_sq failure: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _FLASH_KMEANS_INLINE_XSQ_FALLBACK_WARNED = True
            if _KMEANS_KERNEL_PROFILE_ENABLED:
                timings["kmeans_backend_path"] = "flash"
                timings["inline_xsq_fallback_reason"] = str(exc)
        else:
            if _KMEANS_KERNEL_PROFILE_ENABLED:
                _LAST_KMEANS_TIMINGS.update(timings)
            return out
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid as flash_batch_kmeans_Euclid

    flash_kwargs = {}
    try:
        if "use_heuristic" in inspect.signature(flash_batch_kmeans_Euclid).parameters:
            flash_kwargs["use_heuristic"] = _FLASH_KMEANS_USE_HEURISTIC
    except (TypeError, ValueError):
        flash_kwargs = {}

    (cluster_ids, centroids, n_iters), ms = _profiled_kmeans_ms(
        x.device,
        "kmeans.flash.external_flash_kmeans",
        lambda: flash_batch_kmeans_Euclid(
            x,
            n_clusters=n_clusters,
            max_iters=max_iters,
            tol=tol,
            init_centroids=init_centroids,
            verbose=verbose,
            **flash_kwargs,
        ),
    )
    if _KMEANS_KERNEL_PROFILE_ENABLED:
        timings["flash_kmeans_ms"] = ms
        timings["kmeans_iters_actual"] = int(n_iters)
    cluster_sizes = _cluster_sizes_from_labels(
        cluster_ids,
        n_clusters,
        profile_prefix="cluster_sizes",
        profile_timings=timings,
    )
    if _KMEANS_KERNEL_PROFILE_ENABLED:
        _LAST_KMEANS_TIMINGS.update(timings)
    return cluster_ids, centroids, cluster_sizes, int(n_iters)


def _batch_kmeans_Euclid_n8_cuda_ext(
    x,
    n_clusters,
    max_iters=100,
    tol=1e-4,
    init_centroids=None,
    verbose=False,
):
    """Fixed-iteration D64 Lloyd updates through ``n8_kernels.so``.

    The first N8 call intentionally remains on Flash K-means because it owns
    initialization. The extension is used only when prior centroids provide an
    exact common starting point for an offline parity decision.
    """
    del tol, verbose
    if not x.is_cuda or x.ndim != 3 or x.shape[2] != 64:
        raise RuntimeError("N8 CUDA role clustering requires CUDA X[BH,N,64].")
    if init_centroids is None:
        raise RuntimeError("N8 CUDA role clustering requires initialized centroids.")
    if max_iters <= 0:
        raise RuntimeError("N8 CUDA role clustering requires at least one iteration.")
    from .kernels.n8_extension import role_cluster_step_k64, role_cluster_step_q64

    x = x.float().contiguous()
    centroids = init_centroids.float().contiguous().view(x.shape[0], n_clusters, 64)
    step = role_cluster_step_q64 if n_clusters <= 512 else role_cluster_step_k64
    if n_clusters > 1024:
        raise RuntimeError("N8 CUDA role clustering supports at most 1024 clusters.")
    labels = None
    sizes = None
    for _ in range(int(max_iters)):
        labels, centroids, sizes = step(x, centroids)
    assert labels is not None and sizes is not None
    return labels, centroids, sizes.long(), int(max_iters)


@time_logging_decorator("Level 4 - batch kmeans euclid")
def batch_kmeans_Euclid(
    x,
    n_clusters,
    max_iters=100,
    tol=1e-4,
    init_centroids=None,
    verbose=False,
    backend="cuvs",
):
    """
    Batched KMeans clustering in PyTorch using Euclidean distance.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
        cluster_sizes: (B, n_clusters) LongTensor, number of points per cluster.
        n_iters: actual number of iterations executed (int)
    """
    if max_iters < 0:
        raise RuntimeError(f"max_iters must be non-negative, got {max_iters}.")
    if backend == "flash":
        return _batch_kmeans_Euclid_flash(
            x,
            n_clusters=n_clusters,
            max_iters=max_iters,
            tol=tol,
            init_centroids=init_centroids,
            verbose=verbose,
        )
    if backend == "n8_cuda_ext":
        return _batch_kmeans_Euclid_n8_cuda_ext(
            x,
            n_clusters=n_clusters,
            max_iters=max_iters,
            tol=tol,
            init_centroids=init_centroids,
            verbose=verbose,
        )
    if backend != "cuvs":
        raise RuntimeError(
            f"Unknown Euclidean k-means backend {backend!r}. "
            "Expected 'cuvs', 'flash', or 'n8_cuda_ext'."
        )

    B, N, D = x.shape

    if max_iters == 0:
        return _euclid_assign_only(x, init_centroids, n_clusters, recompute_centroids=False)

    # Pre-compute squared L2 norm of all points (constant during iterations)
    x_sq = (x**2).sum(dim=-1)  # (B, N)

    if init_centroids is None:
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(x, dim=1, index=indices[..., None].expand(-1, -1, D))  # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        centroids_new, center_shift, cluster_ids, cluster_sizes = _euclid_iter_compiled(x, x_sq, centroids)

        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new

    return cluster_ids, centroids, cluster_sizes, it + 1


# batch_kmeans_Euclid = torch.compile(batch_kmeans_Euclid, dynamic=True, mode="reduce-overhead")


def batch_kmeans_Cosine(x, n_clusters, max_iters=100, tol=1e-4, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using Cosine similarity.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
        cluster_sizes: (B, n_clusters) LongTensor, number of points per cluster.
        n_iters: actual number of iterations executed (int)
    """
    B, N, D = x.shape

    # Normalize input vectors for cosine similarity
    x_norm = F.normalize(x, p=2, dim=-1)  # (B, N, D)

    if init_centroids is None:
        # Randomly select initial centers from x_norm
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(x_norm, dim=1, index=indices[..., None].expand(-1, -1, D))  # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)
    centroids = F.normalize(centroids, p=2, dim=-1)  # Ensure centroids are normalized

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _cosine_iter_compiled(x_norm, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    # --- compute cluster sizes ---
    ones = torch.ones_like(cluster_ids, dtype=torch.int64)
    cluster_sizes = torch.zeros(B, n_clusters, dtype=torch.int64, device=x.device)
    cluster_sizes.scatter_add_(1, cluster_ids, ones)

    return cluster_ids, centroids, cluster_sizes, it + 1


def batch_kmeans_Dot(x, n_clusters, max_iters=100, tol=1e-4, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using raw dot-product as similarity.

    """
    B, N, D = x.shape

    if init_centroids is None:
        # Randomly initialize centroids
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(x, dim=1, index=indices[..., None].expand(-1, -1, D))
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _dot_iter_compiled(x, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it} (dot), center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    # --- compute cluster sizes ---
    ones = torch.ones_like(cluster_ids, dtype=torch.int64)
    cluster_sizes = torch.zeros(B, n_clusters, dtype=torch.int64, device=x.device)
    cluster_sizes.scatter_add_(1, cluster_ids, ones)

    return cluster_ids, centroids, cluster_sizes, it + 1


# --- Functions from analyze/kmeans_block_sparse_attention.py (helpers) ---


@time_logging_decorator("Level 4 - permute tensor by labels")
def permute_tensor_by_labels(tensor, labels, dim):
    labels = labels.to(tensor.device)
    sorted_indices = torch.argsort(labels, dim=-1)
    gather_indices = sorted_indices
    for i in range(dim + 1, tensor.dim()):
        gather_indices = gather_indices.unsqueeze(-1)
    expand_shape = list(tensor.shape)
    gather_indices = gather_indices.expand(expand_shape)
    permuted_tensor = torch.gather(tensor, dim, gather_indices)
    return permuted_tensor, sorted_indices


@time_logging_decorator("Level 4 - inverse permutation")
def apply_inverse_permutation(permuted_tensor, sorted_indices, dim):
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    gather_indices = inverse_indices
    for i in range(dim + 1, permuted_tensor.dim()):
        gather_indices = gather_indices.unsqueeze(-1)
    gather_indices = gather_indices.expand(permuted_tensor.shape)
    original_tensor = torch.gather(permuted_tensor, dim, gather_indices)
    return original_tensor


@time_logging_decorator("Level 4 - weighted softmax")
def weighted_softmax(scores, weights):
    input_dtype = scores.dtype
    scores = scores.float()
    weights = weights.float()
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_score)
    weighted_exp = weights * exp_scores
    softmax_out = weighted_exp / torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    return softmax_out.to(input_dtype)


@time_logging_decorator("Level 4 - identify dynamic map")
def identify_dynamic_map(
    query_centroids,
    key_centroids,
    q_cluster_sizes,
    k_cluster_sizes,
    p,
    min_kc_ratio=0,
):
    B, H, qc_num, D = query_centroids.shape
    kc_num = key_centroids.shape[2]
    device = query_centroids.device

    attn_scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (D**0.5)
    k_weights = k_cluster_sizes.unsqueeze(-2).float()

    weighted_attn_probs = weighted_softmax(attn_scores, k_weights)
    sorted_probs, sorted_indices = torch.sort(weighted_attn_probs, dim=-1, descending=True)

    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_indices = cumsum_probs > p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False

    if min_kc_ratio > 0:
        preserve_length = int(min_kc_ratio * kc_num)
        remove_indices[..., :preserve_length] = False

    sorted_clusters_to_keep = ~remove_indices

    dynamic_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
    dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    return dynamic_map


# --- Functions from analyze/dynamic_block_sparse_attention.py ---


def dynamic_block_sparse_fwd_torch(q, k, v, dynamic_map, qc_size, kc_size):
    """
    Computes dynamic block sparse attention using pure PyTorch.

    Args:
        q (torch.Tensor): Query tensor, shape [B, H, S, D].
        k (torch.Tensor): Key tensor, shape [B, H, S, D].
        v (torch.Tensor): Value tensor, shape [B, H, S, D].
        dynamic_map (torch.Tensor): Boolean mask, shape [B, H, qc_num, kc_num].
        qc_size (torch.Tensor): Query block sizes, shape [B, H, qc_num].
        kc_size (torch.Tensor): Key block sizes, shape [B, H, kc_num].

    Returns:
        torch.Tensor: Output tensor, shape [B, H, S, D].
    """
    B, H, S, D = q.shape
    qc_num = qc_size.shape[-1]
    kc_num = kc_size.shape[-1]
    device = q.device
    dtype = q.dtype

    # Ensure sequence lengths match sum of block sizes
    assert S == torch.sum(qc_size[0, 0, :]), "Sum of qc_size must equal S"
    assert S == torch.sum(kc_size[0, 0, :]), "Sum of kc_size must equal S"

    # Precompute cumulative sizes for block indexing
    # Add a 0 at the beginning for easier slicing
    qc_cum_size = torch.cumsum(torch.cat([torch.zeros_like(qc_size[..., :1]), qc_size], dim=-1), dim=-1)
    kc_cum_size = torch.cumsum(torch.cat([torch.zeros_like(kc_size[..., :1]), kc_size], dim=-1), dim=-1)

    out = torch.zeros_like(q)
    scale = D**-0.5

    # Naive implementation: Iterate through batch, head, and blocks
    for b in range(B):
        for h in range(H):
            # Precompute start/end indices for this batch/head
            q_starts = qc_cum_size[b, h, :-1]
            q_ends = qc_cum_size[b, h, 1:]
            k_starts = kc_cum_size[b, h, :-1]
            k_ends = kc_cum_size[b, h, 1:]

            # Iterate through query blocks
            for i in range(qc_num):
                q_start, q_end = q_starts[i], q_ends[i]
                q_block = q[b, h, q_start:q_end, :]  # Shape: [qc_i, D]

                if q_block.shape[0] == 0:
                    continue  # Skip empty blocks

                m_i = torch.full((q_block.shape[0], 1), -float("inf"), device=device, dtype=dtype)
                l_i = torch.zeros((q_block.shape[0], 1), device=device, dtype=dtype)
                acc_o_i = torch.zeros_like(q_block)  # Shape: [qc_i, D]

                # Iterate through key/value blocks for the current query block
                for j in range(kc_num):
                    # Check if this block needs computation
                    if dynamic_map[b, h, i, j]:
                        k_start, k_end = k_starts[j], k_ends[j]
                        k_block = k[b, h, k_start:k_end, :]  # Shape: [kc_j, D]
                        v_block = v[b, h, k_start:k_end, :]  # Shape: [kc_j, D]

                        if k_block.shape[0] == 0:
                            continue  # Skip empty blocks

                        # Compute attention scores for the block
                        # QK^T: [qc_i, D] @ [D, kc_j] -> [qc_i, kc_j]
                        s_ij = (q_block @ k_block.transpose(-1, -2)) * scale

                        # --- Online Softmax ---
                        # Find max score per query token in this block
                        m_ij = torch.max(s_ij, dim=-1, keepdim=True)[0]  # Shape: [qc_i, 1]

                        # Update overall max score (m_i)
                        m_new = torch.maximum(m_i, m_ij)  # Shape: [qc_i, 1]

                        # Calculate scaling factors for previous accumulator and current block
                        p_ij = torch.exp(s_ij - m_new)  # Shape: [qc_i, kc_j]
                        exp_m_diff = torch.exp(m_i - m_new)  # Shape: [qc_i, 1]

                        # Update softmax denominator (l_i)
                        l_i = (l_i * exp_m_diff) + torch.sum(p_ij, dim=-1, keepdim=True)  # Shape: [qc_i, 1]

                        # Update output accumulator (acc_o_i)
                        # P_ij @ V_j: [qc_i, kc_j] @ [kc_j, D] -> [qc_i, D]
                        acc_o_i = (acc_o_i * exp_m_diff) + (p_ij @ v_block)  # Shape: [qc_i, D]

                        # Update max score for next iteration
                        m_i = m_new

                # Normalize the accumulated output
                out[b, h, q_start:q_end, :] = acc_o_i / l_i.clamp(min=1e-12)  # Avoid division by zero

    return out


# --- Triton Implementation ---


@triton.jit
def _dynamic_block_sparse_fwd_kernel(
    Q,
    K,
    V,
    Out,
    dynamic_map,
    qc_cum_size,
    kc_cum_size,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_dmap_b,
    stride_dmap_h,
    stride_dmap_qc,
    stride_dmap_kc,
    stride_qcs_b,
    stride_qcs_h,
    stride_qcs_qc,
    stride_kcs_b,
    stride_kcs_h,
    stride_kcs_kc,
    B,
    H,
    S,
    D,
    scale,
    QC_NUM: tl.constexpr,
    KC_NUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel for dynamic block sparse attention.
    Each program computes attention for one query block within a batch/head.
    Processes query block in chunks of BLOCK_M.
    Iterates through key blocks, checking dynamic_map.
    Processes key/value blocks in chunks of BLOCK_N.
    Uses online softmax.
    """
    # --- Grid Calculation ---
    # Each program instance handles one query block for a specific batch and head
    pid = tl.program_id(axis=0)
    B * H * QC_NUM

    # Calculate batch, head, and query block index
    pid_q_block_global = pid  # 0 to B*H*QC_NUM - 1
    # pid_bh = pid // QC_NUM # Deprecated: Causes issues if QC_NUM is not constant across BH
    # pid_q_block_idx = pid % QC_NUM

    # Need to map pid (0.. B*H*QC_NUM-1) back to (b, h, q_block_idx)
    # q_block_idx changes fastest, then h, then b
    q_block_idx = pid_q_block_global % QC_NUM
    pid_h_temp = pid_q_block_global // QC_NUM
    h = pid_h_temp % H
    b = pid_h_temp // H

    # --- Load Q block info (start/end offsets) ---
    qcs_offset = b * stride_qcs_b + h * stride_qcs_h
    q_start_offset = tl.load(qc_cum_size + qcs_offset + q_block_idx * stride_qcs_qc)
    q_end_offset = tl.load(qc_cum_size + qcs_offset + (q_block_idx + 1) * stride_qcs_qc)
    q_block_size = q_end_offset - q_start_offset

    # Early exit if the query block is empty
    if q_block_size == 0:
        return

    # --- Pointers setup ---
    q_ptr_base = Q + b * stride_qb + h * stride_qh + q_start_offset * stride_qs
    k_ptr_base = K + b * stride_kb + h * stride_kh
    v_ptr_base = V + b * stride_vb + h * stride_vh
    out_ptr_base = Out + b * stride_ob + h * stride_oh + q_start_offset * stride_os
    dmap_ptr = dynamic_map + b * stride_dmap_b + h * stride_dmap_h + q_block_idx * stride_dmap_qc
    kcs_ptr = kc_cum_size + b * stride_kcs_b + h * stride_kcs_h

    # --- Iterate over the query block rows in chunks of BLOCK_M ---
    offs_qm = tl.arange(0, BLOCK_M)  # Query block row offsets [0, 1, ..., BLOCK_M-1]
    offs_d = tl.arange(0, BLOCK_D)  # Dimension offsets [0, 1, ..., BLOCK_D-1]

    for q_chunk_start in range(0, q_block_size, BLOCK_M):
        q_chunk_rows = offs_qm + q_chunk_start
        q_rows_mask = q_chunk_rows < q_block_size  # Mask for valid rows in this Q chunk [BLOCK_M]

        # --- Initialize accumulators for this Q chunk ---
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max score
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum of exp(scores - max)
        acc_o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # Accumulated output

        # --- Load Q chunk ---
        q_ptr = q_ptr_base + q_chunk_rows[:, None] * stride_qs + offs_d[None, :]
        # Mask ensures we don't read out of bounds for the query block or dimension D
        mask_q = q_rows_mask[:, None] & (offs_d[None, :] < D)
        q_chunk = tl.load(q_ptr, mask=mask_q, other=0.0)  # Shape: [BLOCK_M, BLOCK_D]

        # --- Inner loop over K blocks (columns in the block sparse map) ---
        for k_block_idx in range(KC_NUM):
            # --- Check dynamic_map: Is this block active? ---
            is_active = tl.load(dmap_ptr + k_block_idx * stride_dmap_kc)
            if is_active:  # Process block only if it's active
                # --- Load K block info (start/end offsets) ---
                k_start_offset = tl.load(kcs_ptr + k_block_idx * stride_kcs_kc)
                k_end_offset = tl.load(kcs_ptr + (k_block_idx + 1) * stride_kcs_kc)
                k_block_size = k_end_offset - k_start_offset

                # Skip if the key block is empty (inside the active block check)
                if k_block_size > 0:
                    k_block_ptr_base = k_ptr_base + k_start_offset * stride_ks
                    v_block_ptr_base = v_ptr_base + k_start_offset * stride_vs

                    # --- Loop over K block chunks (size BLOCK_N) ---
                    offs_kn = tl.arange(0, BLOCK_N)  # Key block row offsets [0, ..., BLOCK_N-1]
                    for k_chunk_start in range(0, k_block_size, BLOCK_N):
                        k_chunk_rows = offs_kn + k_chunk_start
                        k_rows_mask = k_chunk_rows < k_block_size  # Mask for valid rows in this K/V chunk [BLOCK_N]

                        # --- Load K, V chunks ---
                        k_ptr = k_block_ptr_base + k_chunk_rows[:, None] * stride_ks + offs_d[None, :]
                        v_ptr = v_block_ptr_base + k_chunk_rows[:, None] * stride_vs + offs_d[None, :]

                        # Mask ensures we don't read out of bounds for the key block or dimension D
                        mask_kv = k_rows_mask[:, None] & (offs_d[None, :] < D)
                        k_chunk = tl.load(k_ptr, mask=mask_kv, other=0.0)  # Shape: [BLOCK_N, BLOCK_D]
                        v_chunk = tl.load(v_ptr, mask=mask_kv, other=0.0)  # Shape: [BLOCK_N, BLOCK_D]

                        # --- Compute Scores (Attention) ---
                        # QK^T: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
                        s_ij_chunk = tl.dot(q_chunk, k_chunk.T) * scale

                        # IMPORTANT: Mask out scores corresponding to padding in K before max/softmax
                        # Set scores for invalid K elements to -inf
                        s_ij_chunk = tl.where(k_rows_mask[None, :], s_ij_chunk, -float("inf"))
                        # Mask out scores for invalid Q elements as well (although q_chunk elements are 0, avoid potential issues)
                        s_ij_chunk = tl.where(q_rows_mask[:, None], s_ij_chunk, -float("inf"))

                        # --- Online Softmax Update ---
                        # Current max for this Q-K chunk interaction
                        m_ij_chunk = tl.max(s_ij_chunk, axis=1)  # Shape: [BLOCK_M]

                        # Update overall max (across K chunks seen so far for this Q chunk)
                        m_new = tl.maximum(m_i, m_ij_chunk)  # Shape: [BLOCK_M]

                        # Calculate scaled probabilities P_ij = exp(S_ij - m_new)
                        p_ij_chunk = tl.exp(s_ij_chunk - m_new[:, None])  # Shape: [BLOCK_M, BLOCK_N]
                        # Zero out probabilities for masked K elements before summing
                        p_ij_chunk = tl.where(k_rows_mask[None, :], p_ij_chunk, 0.0)

                        # Calculate scaling factor for previous accumulator state
                        exp_m_diff = tl.exp(m_i - m_new)  # Shape: [BLOCK_M]

                        # Update sum accumulator (denominator L)
                        l_i_chunk = tl.sum(p_ij_chunk, axis=1)  # Sum probabilities for this chunk, shape [BLOCK_M]
                        l_i = (l_i * exp_m_diff) + l_i_chunk  # Shape: [BLOCK_M]

                        # Update output accumulator O
                        # P_ij @ V_j: [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
                        # Ensure p_ij_chunk is the correct dtype for dot product
                        p_ij_chunk_casted = p_ij_chunk.to(V.dtype.element_ty)
                        o_chunk = tl.dot(p_ij_chunk_casted, v_chunk)  # Shape: [BLOCK_M, BLOCK_D]

                        acc_o = (acc_o * exp_m_diff[:, None]) + o_chunk  # Shape: [BLOCK_M, BLOCK_D]

                        # Update max for the next K chunk/block
                        m_i = m_new
            # End of 'if is_active:' block
        # --- End of loop over K blocks ---

        # --- Finalize output for this Q chunk ---
        # Normalize the accumulated output: O = acc_o / l_i
        # Add epsilon to l_i to avoid division by zero
        l_i_safe = tl.where(l_i == 0, 1.0, l_i)  # Avoid 0/0 -> NaN
        o_final_chunk = acc_o / (l_i_safe[:, None])
        o_final_chunk = tl.where(l_i[:, None] == 0, 0.0, o_final_chunk)  # Ensure output is 0 if l_i was 0

        # --- Write output chunk to global memory ---
        out_ptr = out_ptr_base + q_chunk_rows[:, None] * stride_os + offs_d[None, :]
        # Mask ensures we don't write out of bounds for the query block or dimension D
        mask_out = q_rows_mask[:, None] & (offs_d[None, :] < D)
        tl.store(out_ptr, o_final_chunk.to(Out.dtype.element_ty), mask=mask_out)

        # --- (Optional: Write L and M stats if needed) ---
        # Example:
        # l_ptr = L + b * stride_lb + h * stride_lh + (q_start_offset + q_chunk_rows) * stride_ls
        # tl.store(l_ptr, l_i, mask=q_rows_mask)
        # m_ptr = M + ...
        # tl.store(m_ptr, m_i, mask=q_rows_mask)

    # --- End of loop over Q chunks ---


def dynamic_block_sparse_fwd_triton(q, k, v, dynamic_map, qc_size, kc_size):
    """
    Launcher for the Triton dynamic block sparse attention kernel.

    Args:
        q (torch.Tensor): Query tensor, shape [B, H, S, D].
        k (torch.Tensor): Key tensor, shape [B, H, S, D].
        v (torch.Tensor): Value tensor, shape [B, H, S, D].
        dynamic_map (torch.Tensor): Boolean mask, shape [B, H, qc_num, kc_num].
        qc_size (torch.Tensor): Query block sizes, shape [B, H, qc_num].
        kc_size (torch.Tensor): Key block sizes, shape [B, H, kc_num].

    Returns:
        torch.Tensor: Output tensor, shape [B, H, S, D].
    """
    B, H, S, D = q.shape
    qc_num = qc_size.shape[-1]
    kc_num = kc_size.shape[-1]
    dtype = q.dtype

    # Assertions and checks
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    assert dynamic_map.is_cuda and qc_size.is_cuda and kc_size.is_cuda
    assert q.dtype == k.dtype == v.dtype, "Input dtypes must match"
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert D in [16, 32, 64, 128], "Head dimension D must be 16, 32, 64, or 128 for efficient Triton dot"
    # Ensure sequence lengths match sum of block sizes (check on one batch/head for simplicity)
    assert S == torch.sum(qc_size[0, 0, :]), "Sum of qc_size must equal S"
    assert S == torch.sum(kc_size[0, 0, :]), "Sum of kc_size must equal S"
    # Ensure dynamic_map is boolean
    assert dynamic_map.dtype == torch.bool

    # Calculate scale factor (using float32 for stability)
    scale = D**-0.5

    # Precompute cumulative sizes (on CPU/GPU, keep on device)
    qc_cum_size = torch.cumsum(torch.cat([torch.zeros_like(qc_size[..., :1]), qc_size], dim=-1), dim=-1).int()
    kc_cum_size = torch.cumsum(torch.cat([torch.zeros_like(kc_size[..., :1]), kc_size], dim=-1), dim=-1).int()

    # Output tensor
    out = torch.empty_like(q)

    # Triton kernel config
    # BLOCK_M/N can be tuned. Larger blocks may increase occupancy but need more shared memory.
    # Let's start with reasonably sized blocks.
    BLOCK_D = D
    if S <= 512:  # Smaller sequence, smaller blocks might be ok
        BLOCK_M = 64
        BLOCK_N = 64
    elif S <= 1024:
        BLOCK_M = 64
        BLOCK_N = 64
    else:  # Larger sequence, potentially larger blocks
        BLOCK_M = 128  # Or keep 64? Test
        BLOCK_N = 64

    # Adjust block size if sequence length is smaller
    BLOCK_M = min(BLOCK_M, S)
    BLOCK_N = min(BLOCK_N, S)

    # Launch grid: One program per query block per batch/head
    grid = (B * H * qc_num,)

    # Call the kernel
    _dynamic_block_sparse_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        dynamic_map,
        qc_cum_size,
        kc_cum_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dynamic_map.stride(0),
        dynamic_map.stride(1),
        dynamic_map.stride(2),
        dynamic_map.stride(3),
        qc_cum_size.stride(0),
        qc_cum_size.stride(1),
        qc_cum_size.stride(2),
        kc_cum_size.stride(0),
        kc_cum_size.stride(1),
        kc_cum_size.stride(2),
        B,
        H,
        S,
        D,
        scale,
        QC_NUM=qc_num,
        KC_NUM=kc_num,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        # num_warps=4 # Can tune this
    )

    return out


@time_logging_decorator("Level 3 - dynamic block sparse fwd flashinfer on GPU")
def dynamic_block_sparse_fwd_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask_map: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    is_cpu: bool = True,
    num_kv_heads: int | None = None,
    return_lse: bool = False,
    plan_cache_key=None,
    plan_cache_token=None,
):
    """
    Launcher for the Flashinfer dynamic block sparse attention kernel.

    Args:
        q (torch.Tensor): Query tensor, shape [B, H_q, S, D].
        k (torch.Tensor): Key tensor, shape [B, H_kv, S, D] (H_kv == H_q when not GQA-native).
        v (torch.Tensor): Value tensor, shape [B, H_kv, S, D].
        block_mask_map (torch.Tensor): Boolean mask, shape [B, H_kv, qc_num, kc_num].
        block_row_sz (torch.Tensor): Query block sizes, shape [B, H_kv, qc_num].
        block_col_sz (torch.Tensor): Key block sizes, shape [B, H_kv, kc_num].
        is_cpu (bool): Whether mask/sizes are on CPU.
        num_kv_heads (int | None): When set, enables GQA-native mode where
            Q has H_q heads and K/V have H_kv < H_q heads. Block metadata
            is indexed by H_kv. FlashInfer broadcasts within each GQA group.
        return_lse (bool): When True, also return the natural-log LSE of the
            attention logits per query row, shape [B, H_q, S] (float32).
        plan_cache_key: Optional stable, hashable owner key. When supplied, a
            dedicated wrapper retains its current FlashInfer plan.
        plan_cache_token: Identity/version of the current block topology. A
            matching token skips ``wrapper.plan``; a new token replans.
    """
    B, H_q, S, D = q.shape
    lse_scale = _flashinfer_lse_natural_scale(q.device) if return_lse else 1.0
    if num_kv_heads is not None:
        H_kv = num_kv_heads
        assert k.shape[1] == H_kv and v.shape[1] == H_kv, (
            f"GQA-native: K/V must have {H_kv} heads, got K={k.shape[1]}, V={v.shape[1]}"
        )
        assert H_q % H_kv == 0, f"Q heads ({H_q}) must be divisible by KV heads ({H_kv})"
        if not hasattr(dynamic_block_sparse_fwd_flashinfer, "_gqa_native_printed"):
            print(f"[FI-GQA] True GQA-native: Q={list(q.shape)} K={list(k.shape)}")
            print(
                f"  mask={list(block_mask_map.shape)} block_row={list(block_row_sz.shape)} block_col={list(block_col_sz.shape)}"
            )
            print(f"  -> will plan/run with num_qo={B * H_q} num_kv={B * H_kv} (GQA-native)")
            dynamic_block_sparse_fwd_flashinfer._gqa_native_printed = True
    else:
        H_kv = H_q
        assert k.shape[1] == H_q and v.shape[1] == H_q

    qc_num = block_row_sz.shape[-1]
    kc_num = block_col_sz.shape[-1]
    assert block_mask_map.shape == (B, H_kv, qc_num, kc_num)

    assert (
        all(t.device == torch.device("cpu") for t in [block_mask_map, block_row_sz, block_col_sz]) if is_cpu else True
    )

    if _FLASHINFER_RUNTIME_ASSERTS_ENABLED:
        assert torch.all(block_col_sz.sum(dim=2) == block_col_sz.sum(dim=2)[0, 0])
        assert torch.all(block_row_sz.sum(dim=2) == block_row_sz.sum(dim=2)[0, 0])

    _LAST_FLASHINFER_TIMINGS.clear()

    with time_logging_decorator("Level 4 - Planning"):
        if plan_cache_key is not None:
            if plan_cache_token is None:
                raise ValueError(
                    "plan_cache_token is required when plan_cache_key is set."
                )
            if _FLASHINFER_KERNEL_PROFILE_ENABLED:
                (plan_entry, wrapper_cache_hit), setup_ms = _profiled_wall_ms(
                    q.device,
                    lambda: _get_flashinfer_plan_wrapper(q.device, plan_cache_key),
                )
            else:
                plan_entry, wrapper_cache_hit = _get_flashinfer_plan_wrapper(
                    q.device, plan_cache_key
                )
                setup_ms = 0.0
            wrapper_pair = plan_entry["wrapper_pair"]
            workspace_cached = True
            workspace_cache_hit = wrapper_cache_hit
            plan_cache_hit = plan_entry["plan_token"] == plan_cache_token
        else:
            plan_entry = None
            plan_cache_hit = False
            if _FLASHINFER_KERNEL_PROFILE_ENABLED:
                (wrapper_pair, workspace_cached, workspace_cache_hit), setup_ms = _profiled_wall_ms(
                    q.device,
                    lambda: _get_flashinfer_wrapper(q.device),
                )
            else:
                wrapper_pair, workspace_cached, workspace_cache_hit = _get_flashinfer_wrapper(q.device)
                setup_ms = 0.0
        wrapper, _ = wrapper_pair

        q = q.reshape(B * H_q, S, D)
        k = k.reshape(B * H_kv, S, D)
        v = v.reshape(B * H_kv, S, D)
        block_mask_map = block_mask_map.reshape(B * H_kv, qc_num, kc_num)
        block_row_sz = block_row_sz.reshape(B * H_kv, qc_num)
        block_col_sz = block_col_sz.reshape(B * H_kv, kc_num)

        def plan_wrapper():
            with flashinfer_patch_enabled():
                wrapper.plan(
                    block_mask_map=block_mask_map,
                    block_row_sz=block_row_sz,
                    block_col_sz=block_col_sz,
                    num_qo_heads=B * H_q,
                    num_kv_heads=B * H_kv,
                    head_dim=D,
                    q_data_type=q.dtype,
                    kv_data_type=k.dtype,
                )

        if plan_cache_hit:
            planning_ms = 0.0
        elif _FLASHINFER_KERNEL_PROFILE_ENABLED:
            _, planning_ms = _profiled_wall_ms(q.device, plan_wrapper)
        else:
            plan_wrapper()
            planning_ms = 0.0
        if plan_entry is not None and not plan_cache_hit:
            plan_entry["plan_token"] = plan_cache_token

        if not hasattr(dynamic_block_sparse_fwd_flashinfer, "_backend_printed"):
            resolved_backend = getattr(wrapper, "_backend", "unknown")
            print(
                f"[FI-BACKEND] flashinfer=={flashinfer.__version__} "
                f"requested={_FLASHINFER_BACKEND!r} resolved={resolved_backend!r} "
                f"device={torch.cuda.get_device_name(q.device) if q.is_cuda else 'cpu'}"
            )
            if os.getenv("SPARSEPR_FLASHINFER_ASSERT_FA3", "0") == "1":
                assert resolved_backend == "fa3", (
                    f"Expected FlashInfer fa3 backend on this device, got {resolved_backend!r}. "
                    f"Set SPARSEPR_COSMOS3_FLASHINFER_BACKEND=fa3 to force it."
                )
            dynamic_block_sparse_fwd_flashinfer._backend_printed = True

    with time_logging_decorator("Level 4 - Running"):
        lse = None
        if return_lse:
            if _FLASHINFER_KERNEL_PROFILE_ENABLED:
                (o, lse), running_ms = _profiled_wall_ms(q.device, lambda: wrapper.run(q, k, v, return_lse=True))
            else:
                o, lse = wrapper.run(q, k, v, return_lse=True)
                running_ms = 0.0
            lse = lse.reshape(B, H_q, S).float()
            if lse_scale != 1.0:
                lse = lse * lse_scale
        else:
            if _FLASHINFER_KERNEL_PROFILE_ENABLED:
                o, running_ms = _profiled_wall_ms(q.device, lambda: wrapper.run(q, k, v))
            else:
                o = wrapper.run(q, k, v)
                running_ms = 0.0
        o = o.reshape(B, H_q, S, D)
    if _FLASHINFER_KERNEL_PROFILE_ENABLED:
        _LAST_FLASHINFER_TIMINGS.update(
            {
                "flashinfer_workspace_cached": bool(workspace_cached),
                "flashinfer_workspace_cache_hit": bool(workspace_cache_hit),
                "flashinfer_plan_cache_enabled": plan_cache_key is not None,
                "flashinfer_plan_cache_hit": bool(plan_cache_hit),
                "flashinfer_setup_ms": setup_ms,
                "flashinfer_planning_ms": planning_ms,
                "flashinfer_running_ms": running_ms,
                "flashinfer_total_ms": setup_ms + planning_ms + running_ms,
                "flashinfer_bh": int(B * H_q),
                "flashinfer_bh_kv": int(B * H_kv),
                "flashinfer_gqa_native": num_kv_heads is not None,
                "flashinfer_gqa_expanded_back": False,
                "flashinfer_tokens": int(S),
                "flashinfer_head_dim": int(D),
                "flashinfer_q_clusters": int(qc_num),
                "flashinfer_k_clusters": int(kc_num),
            }
        )
    if return_lse:
        return o, lse
    return o


# ---------------- Batch wrapper for cuVS KMeans -----------------


def batch_kmeans_rapidai(x, n_clusters, max_iters=100, tol=1e-4, init_centroids=None, verbose=False):
    """Batched K-Means using RAPIDS cuVS implementation.

    Args:
        x (Tensor): (B, N, D)  float32 tensor on CUDA.
        n_clusters (int): K.
        max_iters (int): maximum iterations.
        tol (float): tolerance.
        init_centroids (Tensor|None): optional initial centroids (B,K,D) float32.
        verbose (bool): print per-batch info.

    Returns:
        cluster_ids  (B, N) LongTensor
        centroids    (B, K, D) float32
        cluster_sizes (B, K)  LongTensor
        n_iters_list (List[int])  iterations per batch
    """
    B, N, D = x.shape
    if init_centroids is not None:
        assert init_centroids.shape == (B, n_clusters, D)

    cluster_ids_list = []
    centroids_list = []
    # cluster_sizes_list = []
    n_iters_list = []

    x_float = x.float()
    if init_centroids is not None:
        init_centroids_float = init_centroids.float()

    for b in range(B):
        xb = x_float[b]
        if init_centroids is None:
            centroids_init_b = None
            init_method = "KMeansPlusPlus"
        else:
            centroids_init_b = init_centroids_float[b]
            init_method = "Array"
        labels_b, centroids_b, n_iter_b = kmeans_rapidai(
            xb, n_clusters, max_iter=max_iters, tol=tol, init_method=init_method, centroids_init=centroids_init_b
        )

        cluster_ids_list.append(labels_b.to(torch.int64))  # (N,)
        centroids_list.append(centroids_b)
        # cluster_sizes_b = torch.bincount(labels_b, minlength=n_clusters).to(torch.int64)
        # cluster_sizes_list.append(cluster_sizes_b)
        # n_iters_list.append(n_iter_b)
        # if verbose:
        #     print(f"Batch {b}: iters={n_iter_b}, cluster sizes min={cluster_sizes_b.min().item()} max={cluster_sizes_b.max().item()}")

    cluster_ids = torch.stack(cluster_ids_list, dim=0)  # (B,N)
    centroids = torch.stack(centroids_list, dim=0).to(x.dtype)  # (B,K,D)
    # cluster_sizes = torch.stack(cluster_sizes_list, dim=0)  # (B,K)
    # --- compute cluster sizes ---
    ones = torch.ones_like(cluster_ids, dtype=torch.int64)
    cluster_sizes = torch.zeros(B, n_clusters, dtype=torch.int64, device=x.device)
    cluster_sizes.scatter_add_(1, cluster_ids, ones)

    return cluster_ids, centroids, cluster_sizes, n_iters_list
