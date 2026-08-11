import torch

import triton
import triton.language as tl


@triton.jit
def _attention_mass_score_kernel(
    Q_CENTROIDS,
    K_SORTED,
    K_OFFSETS,
    SCORES,
    Q_CLUSTERS: tl.constexpr,
    K_CLUSTERS: tl.constexpr,
    SORTED_HEAD_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Exact per-cluster log attention numerator at each query centroid."""
    pid_q = tl.program_id(0)
    pid_kc = tl.program_id(1)
    pid_qh = tl.program_id(2)
    pid_kh = pid_qh // GQA_GROUP

    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_idx = tl.arange(0, DIM)
    n_idx = tl.arange(0, BLOCK_N)
    q_mask = q_idx < Q_CLUSTERS
    q = tl.load(
        Q_CENTROIDS
        + (pid_qh * Q_CLUSTERS + q_idx[:, None]) * DIM
        + d_idx[None, :],
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    start = tl.load(
        K_OFFSETS + pid_kh * (K_CLUSTERS + 1) + pid_kc
    ).to(tl.int32)
    end = tl.load(
        K_OFFSETS + pid_kh * (K_CLUSTERS + 1) + pid_kc + 1
    ).to(tl.int32)
    scale = 1.0 / tl.sqrt(tl.full((), DIM, tl.float32))
    local_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    exp_sum = tl.zeros((BLOCK_Q,), tl.float32)

    token_start = start
    while token_start < end:
        token_idx = token_start + n_idx
        token_mask = token_idx < end
        k = tl.load(
            K_SORTED
            + (
                pid_kh * SORTED_HEAD_STRIDE
                + token_idx[:, None]
            )
            * DIM
            + d_idx[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        logits = tl.dot(q, tl.trans(k), input_precision="tf32") * scale
        masked_logits = tl.where(
            q_mask[:, None] & token_mask[None, :],
            logits,
            -float("inf"),
        )
        chunk_max = tl.max(masked_logits, axis=1)
        new_max = tl.maximum(local_max, chunk_max)
        exp_sum = exp_sum * tl.exp(local_max - new_max)
        exp_sum += tl.sum(
            tl.where(
                q_mask[:, None] & token_mask[None, :],
                tl.exp(logits - new_max[:, None]),
                0.0,
            ),
            axis=1,
        )
        local_max = new_max
        token_start += BLOCK_N

    score = tl.log(tl.maximum(exp_sum, 1.0e-30)) + local_max
    score = tl.where(q_mask & (end > start), score, -float("inf"))
    tl.store(
        SCORES
        + (pid_qh * Q_CLUSTERS + q_idx) * K_CLUSTERS
        + pid_kc,
        score,
        mask=q_mask,
    )


@triton.jit
def _svg_ear_value_error_kernel(
    Q_CENTROIDS,
    K_SORTED,
    V_SORTED,
    K_CENTROIDS,
    V_CENTROIDS,
    VAR_K_DIAG,
    K_OFFSETS,
    SCORES,
    Q_CLUSTERS: tl.constexpr,
    K_CLUSTERS: tl.constexpr,
    TOKENS: tl.constexpr,
    SORTED_HEAD_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    JENSEN_VAR_CAP: tl.constexpr,
    USE_JENSEN: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kc = tl.program_id(1)
    pid_qh = tl.program_id(2)
    pid_kh = pid_qh // GQA_GROUP

    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_idx = tl.arange(0, DIM)
    n_idx = tl.arange(0, BLOCK_N)
    q_mask = q_idx < Q_CLUSTERS

    q_ptrs = Q_CENTROIDS + (pid_qh * Q_CLUSTERS + q_idx[:, None]) * DIM + d_idx[None, :]
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    kc_ptrs = K_CENTROIDS + (pid_kh * K_CLUSTERS + pid_kc) * DIM + d_idx
    vc_ptrs = V_CENTROIDS + (pid_kh * K_CLUSTERS + pid_kc) * DIM + d_idx
    kc = tl.load(kc_ptrs).to(tl.float32)
    vc = tl.load(vc_ptrs).to(tl.float32)

    scale = 1.0 / tl.sqrt(tl.full((), DIM, tl.float32))
    centroid_logits = tl.sum(q * kc[None, :], axis=1) * scale
    if USE_JENSEN:
        var_ptrs = VAR_K_DIAG + (pid_kh * K_CLUSTERS + pid_kc) * DIM + d_idx
        var_k = tl.load(var_ptrs).to(tl.float32)
        s2 = tl.sum(q * q * var_k[None, :], axis=1) / tl.full((), DIM, tl.float32)
        s2 = tl.minimum(s2, tl.full((), JENSEN_VAR_CAP, tl.float32))
        centroid_logits += 0.5 * s2
    # Keep the centroid contribution in a running-max coordinate system.
    # The final score is emitted in log space, so it preserves the exact
    # Equation-8 ordering without ever exponentiating an unshifted logit.
    local_max = centroid_logits
    centroid_weight = tl.full((BLOCK_Q,), 1.0, tl.float32)
    centroid_value_norm2 = tl.sum(vc * vc, axis=0)

    start = tl.load(K_OFFSETS + pid_kh * (K_CLUSTERS + 1) + pid_kc).to(tl.int32)
    end = tl.load(K_OFFSETS + pid_kh * (K_CLUSTERS + 1) + pid_kc + 1).to(tl.int32)
    error_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    token_start = start
    while token_start < end:
        token_idx = token_start + n_idx
        token_mask = token_idx < end
        k_ptrs = K_SORTED + (pid_kh * SORTED_HEAD_STRIDE + token_idx[:, None]) * DIM + d_idx[None, :]
        v_ptrs = V_SORTED + (pid_kh * SORTED_HEAD_STRIDE + token_idx[:, None]) * DIM + d_idx[None, :]
        k = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)

        token_logits = tl.dot(q, tl.trans(k), input_precision="tf32") * scale
        masked_token_logits = tl.where(token_mask[None, :], token_logits, -float("inf"))
        chunk_max = tl.max(masked_token_logits, axis=1)
        new_max = tl.maximum(local_max, chunk_max)
        alpha = tl.exp(local_max - new_max)
        centroid_weight *= alpha
        error_sum *= alpha * alpha

        token_weight = tl.where(
            token_mask[None, :],
            tl.exp(token_logits - new_max[:, None]),
            0.0,
        )
        v_norm2 = tl.sum(v * v, axis=1)
        centroid_dot_v = tl.sum(v * vc[None, :], axis=1)

        error = (
            centroid_weight[:, None] * centroid_weight[:, None] * centroid_value_norm2
            + token_weight * token_weight * v_norm2[None, :]
            - 2.0 * token_weight * centroid_weight[:, None] * centroid_dot_v[None, :]
        )
        error = tl.maximum(error, 0.0)
        error_sum += tl.sum(tl.where(token_mask[None, :], error, 0.0), axis=1)
        local_max = new_max
        token_start += BLOCK_N

    cluster_size = tl.maximum(end - start, 1).to(tl.float32)
    score = tl.log(tl.maximum(error_sum / cluster_size, 1.0e-30)) + 2.0 * local_max
    score = tl.where(end > start, score, -float("inf"))
    out_ptrs = SCORES + (pid_qh * Q_CLUSTERS + q_idx) * K_CLUSTERS + pid_kc
    tl.store(out_ptrs, score, mask=q_mask)


@triton.jit
def _svg_ear_grouped_value_error_kernel(
    Q_CENTROIDS,
    K_SORTED,
    V_SORTED,
    K_CENTROIDS,
    V_CENTROIDS,
    K_OFFSETS,
    SCORES,
    Q_CLUSTERS: tl.constexpr,
    K_CLUSTERS: tl.constexpr,
    TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_GQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute every per-Q-head score in a GQA group with one K/V traversal."""
    pid_q = tl.program_id(0)
    pid_kc = tl.program_id(1)
    pid_kh = tl.program_id(2)

    gq = tl.arange(0, BLOCK_GQ)
    group_idx = gq // BLOCK_Q
    q_local = gq - group_idx * BLOCK_Q
    q_idx = pid_q * BLOCK_Q + q_local
    d_idx = tl.arange(0, DIM)
    n_idx = tl.arange(0, BLOCK_N)
    q_mask = (group_idx < GQA_GROUP) & (q_idx < Q_CLUSTERS)
    q_head = pid_kh * GQA_GROUP + group_idx

    q_row = q_head * Q_CLUSTERS + q_idx
    q_ptrs = Q_CENTROIDS + q_row[:, None] * DIM + d_idx[None, :]
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    kc = tl.load(
        K_CENTROIDS + (pid_kh * K_CLUSTERS + pid_kc) * DIM + d_idx
    ).to(tl.float32)
    vc = tl.load(
        V_CENTROIDS + (pid_kh * K_CLUSTERS + pid_kc) * DIM + d_idx
    ).to(tl.float32)

    scale = 1.0 / tl.sqrt(tl.full((), DIM, tl.float32))
    centroid_logits = tl.sum(q * kc[None, :], axis=1) * scale
    local_max = centroid_logits
    centroid_weight = tl.full((BLOCK_GQ,), 1.0, tl.float32)
    centroid_value_norm2 = tl.sum(vc * vc, axis=0)
    start = tl.load(K_OFFSETS + pid_kh * (K_CLUSTERS + 1) + pid_kc).to(tl.int32)
    end = tl.load(K_OFFSETS + pid_kh * (K_CLUSTERS + 1) + pid_kc + 1).to(tl.int32)
    error_sum = tl.zeros((BLOCK_GQ,), dtype=tl.float32)

    token_start = start
    while token_start < end:
        token_idx = token_start + n_idx
        token_mask = token_idx < end
        offsets = (pid_kh * TOKENS + token_idx[:, None]) * DIM + d_idx[None, :]
        k = tl.load(K_SORTED + offsets, mask=token_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V_SORTED + offsets, mask=token_mask[:, None], other=0.0).to(tl.float32)
        token_logits = tl.dot(q, tl.trans(k), input_precision="tf32") * scale
        masked_logits = tl.where(q_mask[:, None] & token_mask[None, :], token_logits, -float("inf"))
        chunk_max = tl.max(masked_logits, axis=1)
        new_max = tl.maximum(local_max, chunk_max)
        alpha = tl.exp(local_max - new_max)
        centroid_weight *= alpha
        error_sum *= alpha * alpha
        token_weight = tl.where(
            q_mask[:, None] & token_mask[None, :],
            tl.exp(token_logits - new_max[:, None]),
            0.0,
        )
        v_norm2 = tl.sum(v * v, axis=1)
        centroid_dot_v = tl.sum(v * vc[None, :], axis=1)
        error = (
            centroid_weight[:, None] * centroid_weight[:, None] * centroid_value_norm2
            + token_weight * token_weight * v_norm2[None, :]
            - 2.0 * token_weight * centroid_weight[:, None] * centroid_dot_v[None, :]
        )
        error_sum += tl.sum(
            tl.where(token_mask[None, :], tl.maximum(error, 0.0), 0.0),
            axis=1,
        )
        local_max = new_max
        token_start += BLOCK_N

    cluster_size = tl.maximum(end - start, 1).to(tl.float32)
    score = tl.log(tl.maximum(error_sum / cluster_size, 1.0e-30)) + 2.0 * local_max
    score = tl.where(q_mask & (end > start), score, -float("inf"))
    tl.store(
        SCORES + (q_head * Q_CLUSTERS + q_idx) * K_CLUSTERS + pid_kc,
        score,
        mask=q_mask,
    )


@triton.jit
def _svg_ear_compensation_kernel(
    Q,
    SPARSE,
    LSE,
    QLABELS,
    DYNAMIC_MAP,
    K_CENTROIDS,
    K_COUNTS,
    V_CENTROIDS,
    OUT,
    TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    Q_CLUSTERS_PLUS_1: tl.constexpr,
    K_CLUSTERS: tl.constexpr,
    N3_REFORMULATION: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_idx = tl.arange(0, DIM)
    k_local = tl.arange(0, BLOCK_K)
    q_mask = q_idx < TOKENS

    q_ptrs = Q + (pid_bh * TOKENS + q_idx[:, None]) * DIM + d_idx[None, :]
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    lse = tl.load(LSE + pid_bh * TOKENS + q_idx, mask=q_mask, other=-float("inf")).to(tl.float32)
    sparse_ptrs = SPARSE + (pid_bh * TOKENS + q_idx[:, None]) * DIM + d_idx[None, :]
    acc = tl.load(sparse_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    denom = tl.full((BLOCK_Q,), 1.0, tl.float32)
    row_max = lse
    qlabels = tl.load(QLABELS + pid_bh * TOKENS + q_idx, mask=q_mask, other=0).to(tl.int32) + 1
    scale = 1.0 / tl.sqrt(tl.full((), DIM, tl.float32))

    for k_start in range(0, K_CLUSTERS, BLOCK_K):
        k_idx = k_start + k_local
        k_mask = k_idx < K_CLUSTERS
        kc_ptrs = K_CENTROIDS + (pid_bh * K_CLUSTERS + k_idx[:, None]) * DIM + d_idx[None, :]
        vc_ptrs = V_CENTROIDS + (pid_bh * K_CLUSTERS + k_idx[:, None]) * DIM + d_idx[None, :]
        kc = tl.load(kc_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        vc = tl.load(vc_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        logits = tl.dot(q, tl.trans(kc), input_precision="tf32") * scale
        counts = tl.load(K_COUNTS + pid_bh * K_CLUSTERS + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        valid = counts > 0.0
        logits += tl.log(tl.maximum(counts, 1.0))[None, :]

        dm_ptrs = DYNAMIC_MAP + (pid_bh * Q_CLUSTERS_PLUS_1 + qlabels[:, None]) * (K_CLUSTERS + 1) + k_idx[None, :] + 1
        selected = tl.load(dm_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=1) != 0
        active = valid[None, :] & q_mask[:, None] & k_mask[None, :]
        if N3_REFORMULATION:
            # N3: all-centroid baseline plus the aggregate selected residual.
            # FlashInfer provides the aggregate selected exact numerator/LSE,
            # so its residual is exact_selected - selected_centroid.
            all_logits = tl.where(active, logits, -float("inf"))
            block_max = tl.max(all_logits, axis=1)
            new_max = tl.maximum(row_max, block_max)
            old_scale = tl.exp(row_max - new_max)
            all_weights = tl.where(active, tl.exp(logits - new_max[:, None]), 0.0)
            selected_weights = tl.where(selected & active, all_weights, 0.0)
            acc = acc * old_scale[:, None]
            acc += tl.dot(all_weights, vc, input_precision="tf32")
            acc -= tl.dot(selected_weights, vc, input_precision="tf32")
            denom = denom * old_scale + tl.sum(all_weights, axis=1) - tl.sum(selected_weights, axis=1)
        else:
            skipped = (~selected) & active
            skipped_logits = tl.where(skipped, logits, -float("inf"))
            block_max = tl.max(skipped_logits, axis=1)
            new_max = tl.maximum(row_max, block_max)
            old_scale = tl.exp(row_max - new_max)
            weights = tl.where(skipped, tl.exp(logits - new_max[:, None]), 0.0)
            acc = acc * old_scale[:, None] + tl.dot(weights, vc, input_precision="tf32")
            denom = denom * old_scale + tl.sum(weights, axis=1)
        row_max = new_max

    out = acc / tl.maximum(denom[:, None], 1.0e-30)
    out_ptrs = OUT + (pid_bh * TOKENS + q_idx[:, None]) * DIM + d_idx[None, :]
    tl.store(out_ptrs, out, mask=q_mask[:, None])


def attention_mass_scores_triton(
    *,
    qcentroids: torch.Tensor,
    k_sorted: torch.Tensor,
    kcluster_sizes: torch.Tensor,
    gqa_group: int,
    block_q: int = 16,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    logical_tokens: int | None = None,
) -> torch.Tensor:
    """Return exact token-key cluster mass scores, up to a rowwise constant."""
    if not qcentroids.is_cuda:
        raise RuntimeError("attention_mass_scores_triton requires CUDA tensors.")
    q_heads, q_clusters, dim = qcentroids.shape
    kv_heads, stored_tokens, k_dim = k_sorted.shape
    tokens = stored_tokens if logical_tokens is None else int(logical_tokens)
    if kcluster_sizes.ndim != 2 or kcluster_sizes.shape[0] != kv_heads:
        raise RuntimeError(
            "Attention-mass cluster sizes must be [KVH,Ck], got "
            f"{tuple(kcluster_sizes.shape)}."
        )
    k_clusters = int(kcluster_sizes.shape[1])
    if dim != 128 or k_dim != dim:
        raise RuntimeError(
            f"Attention-mass routing requires DIM=128, got q={dim}, k={k_dim}."
        )
    if q_heads != kv_heads * gqa_group:
        raise RuntimeError(
            "Attention-mass routing GQA mismatch: "
            f"q_heads={q_heads}, kv_heads={kv_heads}, group={gqa_group}."
        )
    if tokens <= 0 or tokens > stored_tokens:
        raise RuntimeError(
            f"Attention-mass logical tokens {tokens} must be in [1,{stored_tokens}]."
        )
    if block_q not in {8, 16, 32} or block_n not in {16, 32, 64}:
        raise RuntimeError(
            "Attention-mass routing supports BQ={8,16,32}, BN={16,32,64}."
        )
    if num_warps not in {2, 4, 8} or num_stages not in {1, 2, 3}:
        raise RuntimeError(
            "Attention-mass routing supports warps={2,4,8}, stages={1,2,3}."
        )
    offsets = torch.cat(
        (
            torch.zeros(
                (kv_heads, 1),
                device=kcluster_sizes.device,
                dtype=torch.int32,
            ),
            kcluster_sizes.to(torch.int32).cumsum(dim=1),
        ),
        dim=1,
    ).contiguous()
    scores = torch.empty(
        (q_heads, q_clusters, k_clusters),
        device=qcentroids.device,
        dtype=torch.float32,
    )
    _attention_mass_score_kernel[
        (triton.cdiv(q_clusters, block_q), k_clusters, q_heads)
    ](
        qcentroids.contiguous(),
        k_sorted.contiguous(),
        offsets,
        scores,
        Q_CLUSTERS=q_clusters,
        K_CLUSTERS=k_clusters,
        SORTED_HEAD_STRIDE=stored_tokens,
        DIM=dim,
        GQA_GROUP=gqa_group,
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return scores


def svg_ear_value_error_scores_triton(
    *,
    qcentroids: torch.Tensor,
    k_sorted: torch.Tensor,
    v_sorted: torch.Tensor,
    kcentroids: torch.Tensor,
    vcentroids: torch.Tensor,
    kcluster_sizes: torch.Tensor,
    gqa_group: int,
    var_k_diag: torch.Tensor | None = None,
    jensen_var_cap: float = 0.0,
    block_q: int = 16,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    logical_tokens: int | None = None,
) -> torch.Tensor:
    if not qcentroids.is_cuda:
        raise RuntimeError("svg_ear_value_error_scores_triton requires CUDA tensors.")
    q_heads, q_clusters, dim = qcentroids.shape
    kv_heads, stored_tokens, k_dim = k_sorted.shape
    tokens = stored_tokens if logical_tokens is None else int(logical_tokens)
    k_clusters = kcentroids.shape[1]
    if dim != 128 or k_dim != dim:
        raise RuntimeError(f"SVG-EAR routing requires DIM=128, got q={dim}, k={k_dim}.")
    if q_heads != kv_heads * gqa_group:
        raise RuntimeError(f"SVG-EAR routing GQA mismatch: q_heads={q_heads}, kv_heads={kv_heads}, group={gqa_group}.")
    if tokens <= 0 or tokens > stored_tokens:
        raise RuntimeError(
            f"SVG-EAR logical token count {tokens} must be in [1,{stored_tokens}]."
        )
    if block_q not in {8, 16, 32} or block_n not in {16, 32, 64}:
        raise RuntimeError("SVG-EAR routing supports BQ={8,16,32}, BN={16,32,64}.")
    if num_warps not in {2, 4, 8} or num_stages not in {1, 2, 3}:
        raise RuntimeError("SVG-EAR routing supports warps={2,4,8}, stages={1,2,3}.")
    use_jensen = jensen_var_cap > 0.0
    if use_jensen:
        if var_k_diag is None:
            raise RuntimeError("Jensen-aware SVG-EAR routing requires var_k_diag.")
        if var_k_diag.shape != (kv_heads, k_clusters, dim):
            raise RuntimeError(
                "SVG-EAR routing variance shape mismatch: "
                f"got {tuple(var_k_diag.shape)}, expected {(kv_heads, k_clusters, dim)}."
            )
        var_k_kernel = var_k_diag.contiguous()
    else:
        var_k_kernel = kcentroids
    offsets = torch.cat(
        [
            torch.zeros((kv_heads, 1), device=kcluster_sizes.device, dtype=torch.int32),
            kcluster_sizes.to(torch.int32).cumsum(dim=1),
        ],
        dim=1,
    ).contiguous()
    scores = torch.empty((q_heads, q_clusters, k_clusters), device=qcentroids.device, dtype=torch.float32)
    grid = (triton.cdiv(q_clusters, block_q), k_clusters, q_heads)
    _svg_ear_value_error_kernel[grid](
        qcentroids.contiguous(),
        k_sorted.contiguous(),
        v_sorted.contiguous(),
        kcentroids.contiguous(),
        vcentroids.contiguous(),
        var_k_kernel,
        offsets,
        scores,
        Q_CLUSTERS=q_clusters,
        K_CLUSTERS=k_clusters,
        TOKENS=tokens,
        SORTED_HEAD_STRIDE=stored_tokens,
        DIM=dim,
        GQA_GROUP=gqa_group,
        JENSEN_VAR_CAP=float(jensen_var_cap),
        USE_JENSEN=use_jensen,
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return scores


def svg_ear_grouped_value_error_scores_triton(
    *,
    qcentroids: torch.Tensor,
    k_sorted: torch.Tensor,
    v_sorted: torch.Tensor,
    kcentroids: torch.Tensor,
    vcentroids: torch.Tensor,
    kcluster_sizes: torch.Tensor,
    gqa_group: int,
    block_q: int = 16,
    block_n: int = 32,
    num_warps: int = 8,
    num_stages: int = 1,
) -> torch.Tensor:
    """Emit per-Q-head scores while sharing each GQA group's K/V traversal."""
    if not qcentroids.is_cuda:
        raise RuntimeError("Grouped SVG-EAR routing requires CUDA tensors.")
    q_heads, q_clusters, dim = qcentroids.shape
    kv_heads, tokens, k_dim = k_sorted.shape
    k_clusters = int(kcentroids.shape[1])
    if dim != 128 or k_dim != dim:
        raise RuntimeError(f"Grouped SVG-EAR requires D=128, got q={dim}, k={k_dim}.")
    if q_heads != kv_heads * gqa_group or gqa_group not in {1, 2, 4, 8}:
        raise RuntimeError(
            f"Grouped SVG-EAR head mismatch q={q_heads}, kv={kv_heads}, G={gqa_group}."
        )
    if block_q not in {8, 16} or block_n not in {16, 32, 64}:
        raise RuntimeError("Grouped SVG-EAR supports BQ={8,16}, BN={16,32,64}.")
    if num_warps not in {4, 8} or num_stages not in {1, 2, 3}:
        raise RuntimeError(
            "Grouped SVG-EAR supports warps={4,8}, stages={1,2,3}."
        )
    offsets = torch.cat(
        (
            torch.zeros((kv_heads, 1), device=kcluster_sizes.device, dtype=torch.int32),
            kcluster_sizes.to(torch.int32).cumsum(dim=1),
        ),
        dim=1,
    ).contiguous()
    scores = torch.empty(
        (q_heads, q_clusters, k_clusters),
        device=qcentroids.device,
        dtype=torch.float32,
    )
    block_gq = gqa_group * block_q
    _svg_ear_grouped_value_error_kernel[
        (triton.cdiv(q_clusters, block_q), k_clusters, kv_heads)
    ](
        qcentroids.contiguous(),
        k_sorted.contiguous(),
        v_sorted.contiguous(),
        kcentroids.contiguous(),
        vcentroids.contiguous(),
        offsets,
        scores,
        Q_CLUSTERS=q_clusters,
        K_CLUSTERS=k_clusters,
        TOKENS=tokens,
        DIM=dim,
        GQA_GROUP=gqa_group,
        BLOCK_Q=block_q,
        BLOCK_GQ=block_gq,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return scores


def svg_ear_compensation_triton(
    *,
    q_flat: torch.Tensor,
    sparse_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    qlabels_gen: torch.Tensor,
    dynamic_map_query: torch.Tensor,
    kcentroids: torch.Tensor,
    kcluster_sizes: torch.Tensor,
    vcentroids: torch.Tensor,
    reformulation: str = "skipped",
    block_q: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    if not q_flat.is_cuda:
        raise RuntimeError("svg_ear_compensation_triton requires CUDA tensors.")
    if reformulation not in {"skipped", "n3_residual"}:
        raise RuntimeError(
            f"SVG-EAR compensation reformulation must be 'skipped' or 'n3_residual', got {reformulation!r}."
        )
    bh, tokens, dim = q_flat.shape
    clusters = kcentroids.shape[1]
    if dim != 128:
        raise RuntimeError(f"SVG-EAR compensation requires DIM=128, got {dim}.")
    if sparse_flat.shape != (bh, tokens, dim):
        raise RuntimeError(
            f"SVG-EAR sparse output shape mismatch: got {tuple(sparse_flat.shape)}, expected {(bh, tokens, dim)}."
        )
    if lse_flat.shape != (bh, tokens):
        raise RuntimeError(f"SVG-EAR LSE shape mismatch: got {tuple(lse_flat.shape)}, expected {(bh, tokens)}.")
    if qlabels_gen.shape != (bh, tokens):
        raise RuntimeError(
            f"SVG-EAR query-label shape mismatch: got {tuple(qlabels_gen.shape)}, expected {(bh, tokens)}."
        )
    if dynamic_map_query.ndim != 3 or dynamic_map_query.shape[0] != bh:
        raise RuntimeError(
            f"SVG-EAR dynamic-map shape mismatch: got {tuple(dynamic_map_query.shape)}, "
            f"expected batch-head dimension {bh}."
        )
    if dynamic_map_query.shape[2] != clusters + 1:
        raise RuntimeError(
            f"SVG-EAR dynamic-map K-cluster mismatch: got {dynamic_map_query.shape[2]}, expected {clusters + 1}."
        )
    expected_stats = (bh, clusters, dim)
    if kcentroids.shape != expected_stats or vcentroids.shape != expected_stats:
        raise RuntimeError(
            f"SVG-EAR centroid shape mismatch: K={tuple(kcentroids.shape)}, "
            f"V={tuple(vcentroids.shape)}, expected={expected_stats}."
        )
    if kcluster_sizes.shape != (bh, clusters):
        raise RuntimeError(
            f"SVG-EAR cluster-size shape mismatch: got {tuple(kcluster_sizes.shape)}, expected {(bh, clusters)}."
        )
    out = torch.empty((bh, tokens, dim), device=q_flat.device, dtype=torch.float32)
    grid = (triton.cdiv(tokens, block_q), bh)
    _svg_ear_compensation_kernel[grid](
        q_flat.contiguous(),
        sparse_flat.contiguous(),
        lse_flat.contiguous(),
        qlabels_gen.contiguous(),
        dynamic_map_query.contiguous(),
        kcentroids.contiguous(),
        kcluster_sizes.contiguous(),
        vcentroids.contiguous(),
        out,
        TOKENS=tokens,
        DIM=dim,
        Q_CLUSTERS_PLUS_1=dynamic_map_query.shape[1],
        K_CLUSTERS=clusters,
        N3_REFORMULATION=reformulation == "n3_residual",
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=1,
    )
    return out
