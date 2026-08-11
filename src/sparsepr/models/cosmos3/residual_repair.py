"""Small pure-PyTorch helpers for current-step reduced-rank residual repair."""

from __future__ import annotations

import torch

_IDENTITY_CACHE: dict[tuple[str, int | None, torch.dtype, int], torch.Tensor] = {}
_TOKEN_IDS_CACHE: dict[tuple[str, int | None, int], torch.Tensor] = {}


def _device_key(device: torch.device) -> tuple[str, int | None]:
    resolved = torch.device(device)
    return resolved.type, resolved.index


def cached_identity(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Reuse the small ridge identity across layers and denoising steps."""
    key = (*_device_key(device), dtype, int(size))
    value = _IDENTITY_CACHE.get(key)
    if value is None:
        value = torch.eye(size, device=device, dtype=dtype)
        _IDENTITY_CACHE[key] = value
    return value


def cached_token_ids(tokens: int, *, device: torch.device) -> torch.Tensor:
    """Reuse the fixed token-index row needed by vectorized probe selection."""
    key = (*_device_key(device), int(tokens))
    value = _TOKEN_IDS_CACHE.get(key)
    if value is None:
        value = torch.arange(tokens, device=device, dtype=torch.long).view(1, -1)
        _TOKEN_IDS_CACHE[key] = value
    return value


def select_role_equal_probes(
    role_features: torch.Tensor,
    role_labels: torch.Tensor,
    *,
    num_clusters: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select representative-first equal-coverage probes independently per head."""
    if role_features.ndim != 3 or role_labels.shape != role_features.shape[:2]:
        raise RuntimeError(
            "Role probe selection expects features [BH,N,R] and labels [BH,N], got "
            f"features={tuple(role_features.shape)}, labels={tuple(role_labels.shape)}."
        )
    batch_heads, tokens, feature_dim = role_features.shape
    count = min(max(1, int(rows)), int(tokens))
    all_rows: list[torch.Tensor] = []
    all_weights: list[torch.Tensor] = []
    token_ids = torch.arange(tokens, device=role_features.device, dtype=torch.long)
    for head in range(batch_heads):
        labels = role_labels[head].long()
        features = role_features[head].float()
        sizes = torch.bincount(labels, minlength=num_clusters)
        sums = torch.zeros(
            (num_clusters, feature_dim),
            device=features.device,
            dtype=torch.float32,
        )
        sums.index_add_(0, labels, features)
        centers = sums / sizes.clamp_min(1).unsqueeze(1)
        distance = (features - centers[labels]).square().sum(dim=1)
        remaining_distance = distance.clone()
        group_order = torch.argsort(sizes, descending=True, stable=True)
        chosen_parts: list[torch.Tensor] = []
        chosen_count = 0
        while chosen_count < count:
            best_distance = torch.full(
                (num_clusters,),
                float("inf"),
                device=features.device,
                dtype=torch.float32,
            )
            best_distance.scatter_reduce_(
                0, labels, remaining_distance, reduce="amin", include_self=True
            )
            is_best = remaining_distance == best_distance[labels]
            candidate_tokens = torch.where(
                is_best,
                token_ids,
                torch.full_like(token_ids, tokens),
            )
            best_index = torch.full(
                (num_clusters,),
                tokens,
                device=features.device,
                dtype=torch.long,
            )
            best_index.scatter_reduce_(
                0, labels, candidate_tokens, reduce="amin", include_self=True
            )
            candidates = best_index[group_order]
            candidates = candidates[candidates < tokens]
            if candidates.numel() == 0:
                break
            take = candidates[: count - chosen_count]
            chosen_parts.append(take)
            remaining_distance[take] = float("inf")
            chosen_count += int(take.numel())
        if chosen_count != count:
            raise RuntimeError(
                f"Role probe selection produced {chosen_count}/{count} rows for head {head}."
            )
        selected = torch.cat(chosen_parts)
        selected_labels = labels[selected]
        sampled = torch.bincount(selected_labels, minlength=num_clusters)
        group_weights = sizes.float() / sampled.clamp_min(1).float()
        weights = group_weights[selected_labels]
        weights = weights / weights.mean().clamp_min(1e-20)
        all_rows.append(selected)
        all_weights.append(weights)
    return torch.stack(all_rows), torch.stack(all_weights)


def select_role_equal_probes_vectorized(
    role_features: torch.Tensor,
    role_labels: torch.Tensor,
    *,
    num_clusters: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized representative-first selection for the N8 M<=C regime.

    N8 requests M=64 probes from Cq=300 role groups, so at most one
    representative is needed from each selected group.  Flattening the head
    dimension into scatter operations removes both the per-head loop and the
    representative-round loop from the production path.
    """
    if role_features.ndim != 3 or role_labels.shape != role_features.shape[:2]:
        raise RuntimeError(
            "Role probe selection expects features [BH,N,R] and labels [BH,N], got "
            f"features={tuple(role_features.shape)}, labels={tuple(role_labels.shape)}."
        )
    batch_heads, tokens, feature_dim = role_features.shape
    count = min(max(1, int(rows)), int(tokens))
    if count > int(num_clusters):
        raise RuntimeError(
            "Vectorized role probes currently require rows <= num_clusters; "
            f"got rows={count}, clusters={num_clusters}."
        )

    labels = role_labels.long()
    features = role_features.float()
    sizes = torch.zeros(
        (batch_heads, num_clusters),
        device=features.device,
        dtype=torch.long,
    )
    sizes.scatter_add_(1, labels, torch.ones_like(labels))
    sums = torch.zeros(
        (batch_heads, num_clusters, feature_dim),
        device=features.device,
        dtype=torch.float32,
    )
    sums.scatter_add_(1, labels.unsqueeze(2).expand(-1, -1, feature_dim), features)
    centers = sums / sizes.clamp_min(1).unsqueeze(2)
    distance = (features - torch.gather(
        centers,
        1,
        labels.unsqueeze(2).expand(-1, -1, feature_dim),
    )).square().sum(dim=2)

    best_distance = torch.full(
        (batch_heads, num_clusters),
        float("inf"),
        device=features.device,
        dtype=torch.float32,
    )
    best_distance.scatter_reduce_(
        1,
        labels,
        distance,
        reduce="amin",
        include_self=True,
    )
    token_ids = cached_token_ids(tokens, device=features.device).expand(batch_heads, -1)
    candidate_tokens = torch.where(
        distance == torch.gather(best_distance, 1, labels),
        token_ids,
        torch.full_like(token_ids, tokens),
    )
    best_index = torch.full(
        (batch_heads, num_clusters),
        tokens,
        device=features.device,
        dtype=torch.long,
    )
    best_index.scatter_reduce_(
        1,
        labels,
        candidate_tokens,
        reduce="amin",
        include_self=True,
    )
    group_order = torch.argsort(sizes, dim=1, descending=True, stable=True)
    selected = torch.gather(best_index, 1, group_order)[:, :count]
    fallback = token_ids[:, :count]
    selected = torch.where(selected < tokens, selected, fallback)
    selected_labels = torch.gather(labels, 1, selected)
    sampled = torch.zeros_like(sizes)
    sampled.scatter_add_(1, selected_labels, torch.ones_like(selected_labels))
    group_weights = sizes.float() / sampled.clamp_min(1).float()
    weights = torch.gather(group_weights, 1, selected_labels)
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-20)
    return selected, weights


def select_first_token_largest_groups(
    role_features: torch.Tensor,
    role_labels: torch.Tensor,
    *,
    num_clusters: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the first token from each of the largest role groups.

    This avoids feature-centroid construction and token-to-centroid distances.
    Each selected row retains the production repair weight for its group size.
    """
    if role_features.ndim != 3 or role_labels.shape != role_features.shape[:2]:
        raise RuntimeError(
            "Role probe selection expects features [BH,N,R] and labels [BH,N], got "
            f"features={tuple(role_features.shape)}, labels={tuple(role_labels.shape)}."
        )
    batch_heads, tokens, _ = role_features.shape
    count = min(max(1, int(rows)), int(tokens))
    if count > int(num_clusters):
        raise RuntimeError(
            "First-token role probes require rows <= num_clusters; "
            f"got rows={count}, clusters={num_clusters}."
        )

    labels = role_labels.long()
    sizes = torch.zeros(
        (batch_heads, num_clusters),
        device=labels.device,
        dtype=torch.long,
    )
    sizes.scatter_add_(1, labels, torch.ones_like(labels))
    token_ids = cached_token_ids(tokens, device=labels.device).expand(batch_heads, -1)
    first = torch.full(
        (batch_heads, num_clusters),
        tokens,
        device=labels.device,
        dtype=torch.long,
    )
    first.scatter_reduce_(1, labels, token_ids, reduce="amin", include_self=True)
    group_order = torch.argsort(sizes, dim=1, descending=True, stable=True)
    selected = torch.gather(first, 1, group_order)[:, :count]
    # Preserve an asynchronous CUDA path. This fallback is relevant only when
    # fewer than ``count`` clusters are non-empty; normal N8 Cq=300/M=64 runs
    # select only valid group representatives.
    selected = torch.where(selected < tokens, selected, token_ids[:, :count])
    selected_labels = torch.gather(labels, 1, selected)
    selected_sizes = torch.gather(sizes, 1, selected_labels).float()
    weights = selected_sizes / selected_sizes.mean(dim=1, keepdim=True).clamp_min(1e-20)
    return selected, weights


def fit_reduced_rank_residual(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    *,
    rank: int,
    ridge: float,
    weights: torch.Tensor | None = None,
    output_basis_backend: str = "full",
    cache_identity: bool = False,
    fit_backend: str = "standard",
    decomposition_backend: str = "eigh",
    basis_source: str = "fitted_residual",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit weighted batched ridge RRR in FP32, independent of ambient autocast.

    ``basis_source='raw_residual'`` extracts the output basis directly from
    centered measured probe residuals. ``'fitted_residual'`` preserves the
    historical full-estimator basis extracted after unrestricted ridge fitting.

    Native Wan generation wraps transformer execution in BF16 autocast.
    Merely calling ``.float()`` on the fit inputs is insufficient there because
    autocast converts subsequent batched GEMMs back to BF16.  The dual solver
    then reaches cuSOLVER with a BF16 Gram matrix, for which Cholesky is not
    implemented.  Keep the complete small RRR fit outside autocast so every
    model adapter gets the intended FP32 numerical path.
    """
    if x_fit.device.type in {"cpu", "cuda"}:
        with torch.autocast(device_type=x_fit.device.type, enabled=False):
            return _fit_reduced_rank_residual_fp32(
                x_fit,
                y_fit,
                rank=rank,
                ridge=ridge,
                weights=weights,
                output_basis_backend=output_basis_backend,
                cache_identity=cache_identity,
                fit_backend=fit_backend,
                decomposition_backend=decomposition_backend,
                basis_source=basis_source,
            )
    return _fit_reduced_rank_residual_fp32(
        x_fit,
        y_fit,
        rank=rank,
        ridge=ridge,
        weights=weights,
        output_basis_backend=output_basis_backend,
        cache_identity=cache_identity,
        fit_backend=fit_backend,
        decomposition_backend=decomposition_backend,
        basis_source=basis_source,
    )


def _fit_reduced_rank_residual_fp32(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    *,
    rank: int,
    ridge: float,
    weights: torch.Tensor | None = None,
    output_basis_backend: str = "full",
    cache_identity: bool = False,
    fit_backend: str = "standard",
    decomposition_backend: str = "eigh",
    basis_source: str = "fitted_residual",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implementation for :func:`fit_reduced_rank_residual` with autocast disabled."""
    if x_fit.ndim != 3 or y_fit.ndim != 3 or x_fit.shape[:2] != y_fit.shape[:2]:
        raise RuntimeError(
            "RRR expects X=[BH,M,P] and Y=[BH,M,D] with matching BH/M, got "
            f"X={x_fit.shape}, Y={y_fit.shape}."
        )
    if rank <= 0 or rank > y_fit.shape[2] or rank > x_fit.shape[1]:
        raise RuntimeError(
            f"Invalid RRR rank={rank} for X={tuple(x_fit.shape)}, Y={tuple(y_fit.shape)}."
        )
    if ridge < 0.0:
        raise RuntimeError(f"RRR ridge must be non-negative, got {ridge}.")
    if output_basis_backend not in {"full", "skinny"}:
        raise RuntimeError(
            "RRR output_basis_backend must be 'full' or 'skinny', got "
            f"{output_basis_backend!r}."
        )
    if fit_backend not in {"standard", "dual_cholesky"}:
        raise RuntimeError(
            "RRR fit_backend must be 'standard' or 'dual_cholesky', got "
            f"{fit_backend!r}."
        )
    if decomposition_backend not in {"eigh", "svd_gesvdj", "svd_gesvd", "triton_top16"}:
        raise RuntimeError(
            "RRR decomposition_backend must be 'eigh', 'svd_gesvdj', "
            f"'svd_gesvd', or 'triton_top16'; got {decomposition_backend!r}."
        )
    if basis_source not in {"fitted_residual", "raw_residual"}:
        raise RuntimeError(
            "RRR basis_source must be 'fitted_residual' or 'raw_residual', got "
            f"{basis_source!r}."
        )
    x = x_fit.float()
    y = y_fit.float()
    if weights is None:
        w = torch.ones((*x.shape[:2], 1), device=x.device, dtype=torch.float32)
        x_mean = x.mean(dim=1, keepdim=True)
        x_scale = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        y_mean = y.mean(dim=1, keepdim=True)
    else:
        if weights.shape != x.shape[:2]:
            raise RuntimeError(
                f"RRR weights must have shape {tuple(x.shape[:2])}, got {tuple(weights.shape)}."
            )
        w = weights.to(device=x.device, dtype=torch.float32).unsqueeze(2)
        w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-20)
        x_mean = (w * x).sum(dim=1, keepdim=True) / w_sum
        x_var = (w * (x - x_mean).square()).sum(dim=1, keepdim=True) / w_sum
        x_scale = x_var.sqrt().clamp_min(1e-5)
        y_mean = (w * y).sum(dim=1, keepdim=True) / w_sum
    x_normalized = (x - x_mean) / x_scale
    y_centered = y - y_mean
    sqrt_w = w.sqrt()
    x_weighted = x_normalized * sqrt_w
    y_weighted = y_centered * sqrt_w
    feature_dim = int(x.shape[2])
    rows = int(x.shape[1])
    dual: torch.Tensor | None = None
    if rows < feature_dim:
        gram = torch.bmm(x_weighted, x_weighted.transpose(1, 2))
        if fit_backend == "dual_cholesky":
            # The ridge system is symmetric positive definite.  Keep the
            # solution in M-row dual space until after the rank-r output basis
            # is known, avoiding the large [BH,P,D] weight and its GEMMs.
            # Only the diagonal changes. Avoid multiplying and adding a dense
            # cached identity in the optimized fixed-shape path.
            regularized_gram = gram
            regularized_gram.diagonal(dim1=1, dim2=2).add_(float(ridge))
            factor, _info = torch.linalg.cholesky_ex(regularized_gram, check_errors=False)
            torch._assert_async(
                (_info == 0).all(),
                "N8 dual Cholesky factorization failed.",
            )
            dual = torch.cholesky_solve(y_weighted, factor)
            # Exact dual identity:
            # sqrt(W) @ fitted = (Xw @ Xw.T) @ dual
            #                    = y_weighted - ridge * dual.
            # This removes two fit GEMMs and the materialized fitted tensor.
            weighted_fitted = torch.add(
                y_weighted,
                dual,
                alpha=-float(ridge),
            )
        else:
            identity_base = (
                cached_identity(rows, device=x.device, dtype=torch.float32)
                if cache_identity
                else torch.eye(rows, device=x.device, dtype=torch.float32)
            )
            identity = identity_base.expand(x.shape[0], -1, -1)
            regularized_gram = gram + float(ridge) * identity
            dual = torch.linalg.solve(regularized_gram, y_weighted)
            weight = torch.bmm(x_weighted.transpose(1, 2), dual)
            fitted = torch.bmm(x_normalized, weight)
    else:
        gram = torch.bmm(x_weighted.transpose(1, 2), x_weighted)
        rhs = torch.bmm(x_weighted.transpose(1, 2), y_weighted)
        identity_base = (
            cached_identity(feature_dim, device=x.device, dtype=torch.float32)
            if cache_identity
            else torch.eye(feature_dim, device=x.device, dtype=torch.float32)
        )
        identity = identity_base.expand(x.shape[0], -1, -1)
        weight = torch.linalg.solve(gram + float(ridge) * identity, rhs)
        fitted = torch.bmm(x_normalized, weight)
    if output_basis_backend == "skinny" and rows < int(y.shape[2]):
        # The nonzero right singular subspace of Z=sqrt(W)@fitted can be
        # recovered from the MxM row Gram Z@Z^T.  N8 therefore diagonalizes
        # 64x64 matrices rather than 128x128 output covariance matrices.
        if basis_source == "raw_residual":
            basis_input = y_weighted
        else:
            if fit_backend != "dual_cholesky" or rows >= feature_dim:
                weighted_fitted = fitted * sqrt_w
            basis_input = weighted_fitted
        basis = reduced_rank_output_basis(
            basis_input,
            rank=int(rank),
            backend=decomposition_backend,
        )
    else:
        basis_values = y_centered if basis_source == "raw_residual" else fitted
        output_gram = torch.bmm(
            basis_values.transpose(1, 2),
            basis_values * w,
        )
        _eigenvalues, eigenvectors = torch.linalg.eigh(output_gram)
        basis = eigenvectors[:, :, -int(rank) :].contiguous()
    if fit_backend == "dual_cholesky" and dual is not None:
        left = torch.bmm(
            x_weighted.transpose(1, 2),
            torch.bmm(dual, basis),
        ).contiguous()
    else:
        left = torch.bmm(weight, basis).contiguous()
    return x_mean, x_scale, y_mean, left, basis


def reduced_rank_output_basis(
    weighted_fitted: torch.Tensor,
    *,
    rank: int,
    backend: str,
) -> torch.Tensor:
    """Return the exact top-r right singular basis of [BH,M,D] fitted values."""
    if weighted_fitted.ndim != 3:
        raise RuntimeError(
            "weighted_fitted must have shape [BH,M,D], got "
            f"{tuple(weighted_fitted.shape)}."
        )
    if rank <= 0 or rank > min(weighted_fitted.shape[1:]):
        raise RuntimeError(
            f"Invalid rank={rank} for weighted_fitted={tuple(weighted_fitted.shape)}."
        )
    if backend == "eigh":
        row_gram = torch.bmm(weighted_fitted, weighted_fitted.transpose(1, 2))
        eigenvalues, left_vectors = torch.linalg.eigh(row_gram)
        left_vectors = left_vectors[:, :, -rank:]
        scales = eigenvalues[:, -rank:].clamp_min(1e-20).rsqrt().unsqueeze(1)
        return (
            torch.bmm(weighted_fitted.transpose(1, 2), left_vectors) * scales
        ).contiguous()
    if backend == "triton_top16":
        if rank != 16:
            raise RuntimeError(f"triton_top16 requires rank=16, got {rank}.")
        try:
            from ...kernels.triton.rrr_eigensolver import top16_right_basis_triton
        except ImportError:
            from ...kernels.triton.rrr_eigensolver import top16_right_basis_triton
        return top16_right_basis_triton(weighted_fitted)
    if backend not in {"svd_gesvdj", "svd_gesvd"}:
        raise RuntimeError(f"Unknown exact decomposition backend {backend!r}.")
    svd_kwargs: dict[str, object] = {"full_matrices": False}
    if weighted_fitted.is_cuda:
        svd_kwargs["driver"] = backend.removeprefix("svd_")
    _u, _s, vh = torch.linalg.svd(weighted_fitted, **svd_kwargs)
    return vh[:, :rank, :].transpose(1, 2).contiguous()


def recover_nonfinite_rrr_state(
    state: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    previous: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None,
) -> tuple[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    torch.Tensor,
    str,
]:
    """Contain a failed current-step RRR solve independently per attention head.

    The fixed-rank output subspace can become numerically rank-deficient for an
    isolated head. Reuse that head's immediately previous finite affine repair
    state while every healthy head keeps the current fit. If no previous state
    exists, use a zero correction for only the failed head.
    """
    if len(state) != 5:
        raise RuntimeError(f"RRR state must contain five tensors, got {len(state)}.")
    batch_heads = int(state[0].shape[0])
    if any(int(tensor.shape[0]) != batch_heads for tensor in state):
        raise RuntimeError("RRR state tensors must share their batch-head dimension.")
    recovered = torch.zeros(
        batch_heads,
        device=state[0].device,
        dtype=torch.bool,
    )
    for tensor in state:
        if not torch.is_floating_point(tensor):
            raise RuntimeError("RRR state tensors must be floating point.")
        recovered |= ~torch.isfinite(tensor).reshape(batch_heads, -1).all(dim=1)
    if previous is None:
        fallback = (
            torch.zeros_like(state[0]),
            torch.ones_like(state[1]),
            torch.zeros_like(state[2]),
            torch.zeros_like(state[3]),
            torch.zeros_like(state[4]),
        )
        policy = "zero_repair_without_previous_state"
    else:
        if len(previous) != len(state) or any(
            old.shape != new.shape
            or old.device != new.device
            or old.dtype != new.dtype
            for old, new in zip(previous, state, strict=True)
        ):
            raise RuntimeError("Previous RRR state does not match the current fit.")
        fallback = previous
        policy = "reuse_previous_finite_repair_state"
    selected = tuple(
        torch.where(
            recovered.reshape(
                batch_heads,
                *((1,) * (current.ndim - 1)),
            ),
            old,
            current,
        )
        for current, old in zip(state, fallback, strict=True)
    )
    return selected, recovered, policy


def predict_reduced_rank_residual(
    x: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply a fitted RRR state to [BH,N,D] features."""
    x_mean, x_scale, y_mean, left, basis = state
    normalized = (x.float() - x_mean) / x_scale
    return torch.bmm(torch.bmm(normalized, left), basis.transpose(1, 2)) + y_mean


def predict_reduced_rank_residual_split(
    base: torch.Tensor,
    role: torch.Tensor | None,
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply output+role RRR without materializing a concatenated feature tensor."""
    x_mean, x_scale, y_mean, left, basis = state
    base_dim = int(base.shape[2])
    rank_coordinates = torch.bmm(
        (base.float() - x_mean[:, :, :base_dim]) / x_scale[:, :, :base_dim],
        left[:, :base_dim],
    )
    if role is not None:
        rank_coordinates = rank_coordinates + torch.bmm(
            (role.float() - x_mean[:, :, base_dim:]) / x_scale[:, :, base_dim:],
            left[:, base_dim:],
        )
    return torch.bmm(rank_coordinates, basis.transpose(1, 2)) + y_mean


def apply_reduced_rank_residual(
    base: torch.Tensor,
    role: torch.Tensor | None,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    y_mean: torch.Tensor,
    left: torch.Tensor,
    basis: torch.Tensor,
    alpha: float,
    norm_cap: float,
) -> torch.Tensor:
    """Compiled-friendly normalization, low-rank apply, cap, and base addition."""
    correction = predict_reduced_rank_residual_split(
        base,
        role,
        (x_mean, x_scale, y_mean, left, basis),
    )
    if norm_cap > 0.0:
        correction_norm = torch.linalg.vector_norm(
            correction, dim=2, keepdim=True
        ).clamp_min(1e-20)
        base_float = base.float()
        base_norm = torch.linalg.vector_norm(base_float, dim=2, keepdim=True)
        correction = correction * (
            float(norm_cap) * base_norm / correction_norm
        ).clamp_max(1.0)
    return (base.float() + float(alpha) * correction).to(base.dtype)
