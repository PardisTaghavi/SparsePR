"""Operator-induced embeddings for Cosmos3 query/key k-means."""

from __future__ import annotations

import os
import warnings

import torch

_ROLE_METRIC_BACKEND = os.environ.get("SPARSEPR_COSMOS3_ROLE_METRIC_BACKEND", "legacy_fp32").strip()
if _ROLE_METRIC_BACKEND not in {"legacy_fp32", "tensorcore_fp32"}:
    raise RuntimeError(
        f"SPARSEPR_COSMOS3_ROLE_METRIC_BACKEND must be 'legacy_fp32' or 'tensorcore_fp32', got {_ROLE_METRIC_BACKEND!r}."
    )

_ORIGINAL_CENTROID_BACKEND = os.environ.get(
    "SPARSEPR_COSMOS3_ORIGINAL_CENTROID_BACKEND", "triton_atomic_fp32"
).strip()
if _ORIGINAL_CENTROID_BACKEND not in {"scatter_fp32", "triton_atomic_fp32"}:
    raise RuntimeError(
        "SPARSEPR_COSMOS3_ORIGINAL_CENTROID_BACKEND must be 'scatter_fp32' or "
        f"'triton_atomic_fp32', got {_ORIGINAL_CENTROID_BACKEND!r}."
    )

_METRIC_EIGH_FALLBACK_WARNED = False


def role_metric_backend() -> str:
    return _ROLE_METRIC_BACKEND


def set_role_metric_backend(backend: str) -> None:
    """Set the role metric backend for offline same-process comparisons."""
    global _ROLE_METRIC_BACKEND
    if backend not in {"legacy_fp32", "tensorcore_fp32"}:
        raise RuntimeError(f"Unknown role metric backend {backend!r}.")
    _ROLE_METRIC_BACKEND = backend


def original_centroid_backend() -> str:
    return _ORIGINAL_CENTROID_BACKEND


def set_original_centroid_backend(backend: str) -> None:
    """Set original-space centroid construction for offline A/B benchmarks."""
    global _ORIGINAL_CENTROID_BACKEND
    if backend not in {"scatter_fp32", "triton_atomic_fp32"}:
        raise RuntimeError(f"Unknown original centroid backend {backend!r}.")
    _ORIGINAL_CENTROID_BACKEND = backend


def deterministic_token_sample(x: torch.Tensor, sample_tokens: int) -> torch.Tensor:
    if x.ndim != 3:
        raise RuntimeError(f"Expected [batch_heads, tokens, dim], got {tuple(x.shape)}.")
    tokens = int(x.shape[1])
    if sample_tokens <= 0 or sample_tokens >= tokens:
        return x
    positions = torch.linspace(0, tokens - 1, steps=sample_tokens, device=x.device)
    return x[:, positions.round().long().unique(sorted=True)]


def covariance(x: torch.Tensor, *, center: bool) -> torch.Tensor:
    """Compute the intended FP32 covariance independently of ambient autocast."""
    if x.device.type in {"cpu", "cuda"}:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return _covariance_fp32(x, center=center)
    return _covariance_fp32(x, center=center)


def _covariance_fp32(x: torch.Tensor, *, center: bool) -> torch.Tensor:
    if _ROLE_METRIC_BACKEND == "tensorcore_fp32" and x.is_cuda and x.dtype in {torch.bfloat16, torch.float16}:
        # BF16/FP16 values are the original model values. Their products are
        # accumulated into FP32 directly by GEMM, avoiding a full token-sized
        # FP32 materialization while preserving an FP32 covariance output.
        try:
            second = torch.bmm(
                x.transpose(1, 2),
                x,
                out_dtype=torch.float32,
            ) / max(int(x.shape[1]), 1)
        except TypeError:
            # Older PyTorch builds do not expose bmm(out_dtype=...).
            second = torch.bmm(x.float().transpose(1, 2), x.float()) / max(int(x.shape[1]), 1)
        if not center:
            return second
        mean = x.mean(dim=1, dtype=torch.float32)
        return second - mean.unsqueeze(2) * mean.unsqueeze(1)
    values = x.float()
    if center:
        values = values - values.mean(dim=1, keepdim=True)
    return torch.bmm(values.transpose(1, 2), values) / max(int(values.shape[1]), 1)


def metric_factor(metric: torch.Tensor, rank: int) -> torch.Tensor:
    """Factor a role metric in FP32 independently of ambient autocast."""
    if metric.device.type in {"cpu", "cuda"}:
        with torch.autocast(device_type=metric.device.type, enabled=False):
            return _metric_factor_fp32(metric, rank)
    return _metric_factor_fp32(metric, rank)


def _metric_factor_fp32(metric: torch.Tensor, rank: int) -> torch.Tensor:
    if metric.ndim != 3 or metric.shape[1] != metric.shape[2]:
        raise RuntimeError(f"Expected batched square metrics, got {tuple(metric.shape)}.")
    if rank <= 0 or rank > metric.shape[-1]:
        raise RuntimeError(f"rank must be in [1, {metric.shape[-1]}], got {rank}.")
    metric_fp32 = metric.float()
    if not torch.isfinite(metric_fp32).all():
        raise RuntimeError("Role metric contains non-finite values before factorization.")
    # Covariance metrics are symmetric in exact arithmetic. Explicitly remove
    # small GEMM roundoff asymmetry before passing them to a batched solver.
    metric_fp32 = 0.5 * (metric_fp32 + metric_fp32.transpose(1, 2))
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(metric_fp32)
        eigenvalues = eigenvalues[:, -rank:].clamp_min(0.0)
        eigenvectors = eigenvectors[:, :, -rank:]
    except torch.linalg.LinAlgError:
        # CUDA's divide-and-conquer symmetric eigensolver can rarely fail to
        # converge for low-rank covariance batches with repeated eigenvalues.
        # The more conservative QR SVD driver returns the same dominant
        # subspace for these positive-semidefinite metrics. This path runs only
        # when EIGH has already failed, normally once when a WAN expert starts.
        global _METRIC_EIGH_FALLBACK_WARNED
        if not _METRIC_EIGH_FALLBACK_WARNED:
            warnings.warn(
                "torch.linalg.eigh did not converge for a role metric; "
                "using the robust SVD metric-factor fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            _METRIC_EIGH_FALLBACK_WARNED = True
        if metric_fp32.is_cuda:
            left_vectors, _, _ = torch.linalg.svd(
                metric_fp32, full_matrices=False, driver="gesvd"
            )
        else:
            left_vectors, _, _ = torch.linalg.svd(
                metric_fp32, full_matrices=False
            )
        eigenvectors = left_vectors[:, :, :rank]
        # Use signed Rayleigh quotients rather than singular values so tiny
        # negative numerical modes retain the same clamp-to-zero semantics as
        # the EIGH path.
        eigenvalues = torch.einsum(
            "bdr,bde,ber->br",
            eigenvectors,
            metric_fp32,
            eigenvectors,
        ).clamp_min(0.0)
    normalizer = eigenvalues.mean(dim=1, keepdim=True).clamp_min(1e-12)
    return eigenvectors * torch.sqrt(eigenvalues / normalizer).unsqueeze(1)


def cholesky_with_device_fallback(
    matrix: torch.Tensor,
    scale: torch.Tensor,
    *,
    relative_ridge: float,
    fallback_relative_ridge: float = 1e-2,
    return_failure_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Batched Cholesky with a synchronized-host-free ridge fallback.

    Both candidates are launched so selection can remain entirely on device.
    Well-conditioned batches retain the requested ridge; only batches whose
    primary ``cholesky_ex`` reports failure use the stronger fallback.
    """

    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise RuntimeError(f"Expected batched square matrices, got {tuple(matrix.shape)}.")
    if scale.shape != (matrix.shape[0],):
        raise RuntimeError(
            f"Cholesky scale shape mismatch: got {tuple(scale.shape)}, "
            f"expected {(matrix.shape[0],)}."
        )
    if relative_ridge <= 0.0 or fallback_relative_ridge <= 0.0:
        raise RuntimeError("Cholesky ridge values must be positive.")
    fallback_relative_ridge = max(fallback_relative_ridge, relative_ridge)
    identity = torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    ).expand(matrix.shape[0], -1, -1)
    primary_matrix = matrix + (
        scale * float(relative_ridge)
    )[:, None, None] * identity
    fallback_matrix = matrix + (
        scale * float(fallback_relative_ridge)
    )[:, None, None] * identity
    primary, primary_info = torch.linalg.cholesky_ex(
        primary_matrix, check_errors=False
    )
    fallback, fallback_info = torch.linalg.cholesky_ex(
        fallback_matrix, check_errors=False
    )
    # A 1e-2 relative shift is deliberately much larger than FP32 Gram/Rayleigh
    # roundoff. If it still fails, the matrix contains non-finite or materially
    # indefinite values and should not be silently propagated.
    fallback_failed = fallback_info != 0
    if (
        not matrix.is_cuda
        and not return_failure_mask
        and bool(fallback_failed.any().item())
    ):
        raise RuntimeError("Cholesky fallback failed for a materially invalid matrix.")
    use_primary = (primary_info == 0)[:, None, None]
    use_fallback = (~use_primary) & (~fallback_failed[:, None, None])
    # CUDA callers recover failed heads from their last finite role factor.
    # Keep this helper device-only and finite until that per-head selection is
    # applied; returning a failed cholesky_ex output would otherwise introduce
    # NaNs before the caller can identify the affected batch.
    safe_placeholder = identity
    selected = torch.where(
        use_primary,
        primary,
        torch.where(use_fallback, fallback, safe_placeholder),
    )
    if return_failure_mask:
        return selected, fallback_failed
    return selected


def warm_subspace_metric_factor(
    metric: torch.Tensor,
    previous_factor: torch.Tensor,
    rank: int,
    *,
    power_iters: int = 1,
    orthogonalization: str = "each_qr",
    ritz_mode: str = "full",
    rayleigh_ridge: float = 1e-6,
    cholesky_qr_ridge: float = 1e-4,
    return_basis: bool = False,
    return_recovery_mask: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Update the warm role factor in FP32 independently of ambient autocast."""
    if metric.device.type in {"cpu", "cuda"}:
        with torch.autocast(device_type=metric.device.type, enabled=False):
            return _warm_subspace_metric_factor_fp32(
                metric,
                previous_factor,
                rank,
                power_iters=power_iters,
                orthogonalization=orthogonalization,
                ritz_mode=ritz_mode,
                rayleigh_ridge=rayleigh_ridge,
                cholesky_qr_ridge=cholesky_qr_ridge,
                return_basis=return_basis,
                return_recovery_mask=return_recovery_mask,
            )
    return _warm_subspace_metric_factor_fp32(
        metric,
        previous_factor,
        rank,
        power_iters=power_iters,
        orthogonalization=orthogonalization,
        ritz_mode=ritz_mode,
        rayleigh_ridge=rayleigh_ridge,
        cholesky_qr_ridge=cholesky_qr_ridge,
        return_basis=return_basis,
        return_recovery_mask=return_recovery_mask,
    )


def _warm_subspace_metric_factor_fp32(
    metric: torch.Tensor,
    previous_factor: torch.Tensor,
    rank: int,
    *,
    power_iters: int = 1,
    orthogonalization: str = "each_qr",
    ritz_mode: str = "full",
    rayleigh_ridge: float = 1e-6,
    cholesky_qr_ridge: float = 1e-4,
    return_basis: bool = False,
    return_recovery_mask: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Update a previous top-r metric factor with current-metric subspace iteration.

    The metric and all orthogonalization/Rayleigh--Ritz operations remain FP32.
    Unlike stale-basis reuse, this updates the subspace on every call.  The only
    eigendecomposition is the reduced ``rank x rank`` projected metric.
    """
    if metric.ndim != 3 or metric.shape[1] != metric.shape[2]:
        raise RuntimeError(f"Expected batched square metrics, got {tuple(metric.shape)}.")
    if rank <= 0 or rank > metric.shape[-1]:
        raise RuntimeError(f"rank must be in [1, {metric.shape[-1]}], got {rank}.")
    expected = (metric.shape[0], metric.shape[1], rank)
    if previous_factor.shape != expected:
        raise RuntimeError(
            f"Previous factor shape mismatch: got {tuple(previous_factor.shape)}, expected {expected}."
        )
    if power_iters < 1:
        raise RuntimeError(f"power_iters must be positive, got {power_iters}.")
    if rayleigh_ridge <= 0.0:
        raise RuntimeError(f"rayleigh_ridge must be positive, got {rayleigh_ridge}.")
    if cholesky_qr_ridge <= 0.0:
        raise RuntimeError(
            f"cholesky_qr_ridge must be positive, got {cholesky_qr_ridge}."
        )
    if orthogonalization not in {
        "each_qr",
        "final_qr",
        "cholesky_qr",
        "each_cholesky_qr",
        "each_cholesky_qr2",
    }:
        raise RuntimeError(f"Unknown subspace orthogonalization {orthogonalization!r}.")
    if ritz_mode not in {
        "full",
        "rayleigh_cholesky",
        "diagonal",
        "previous_scale",
    }:
        raise RuntimeError(f"Unknown Rayleigh--Ritz mode {ritz_mode!r}.")

    metric_f = metric.float()
    previous_scales = torch.linalg.vector_norm(
        previous_factor.float(), dim=1
    ).clamp_min(1e-12)
    diagonal_values = None
    recovery_mask = torch.zeros(
        metric_f.shape[0], device=metric_f.device, dtype=torch.bool
    )

    def cholesky_qr(
        x: torch.Tensor,
        *,
        relative_ridge: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize first so one relative ridge works across layers and metrics.
        normalized = x / torch.linalg.vector_norm(
            x, dim=1, keepdim=True
        ).clamp_min(1e-12)
        gram = torch.bmm(normalized.transpose(1, 2), normalized)
        gram = 0.5 * (gram + gram.transpose(1, 2))
        jitter = (
            gram.diagonal(dim1=1, dim2=2)
            .mean(dim=1, keepdim=True)
            .clamp_min(1e-12)
            * relative_ridge
        )
        gram.diagonal(dim1=1, dim2=2).add_(jitter)
        chol, info = torch.linalg.cholesky_ex(
            gram,
            check_errors=False,
        )
        failed = info != 0
        identity = torch.eye(
            chol.shape[-1], device=chol.device, dtype=chol.dtype
        ).expand_as(chol)
        chol = torch.where(failed[:, None, None], identity, chol)
        orthogonalized = torch.linalg.solve_triangular(
            chol,
            normalized.transpose(1, 2),
            upper=False,
        ).transpose(1, 2)
        return orthogonalized, failed

    if orthogonalization == "each_qr":
        basis, _ = torch.linalg.qr(previous_factor.float(), mode="reduced")
        for iteration in range(power_iters):
            applied = torch.bmm(metric_f, basis)
            if ritz_mode == "diagonal" and iteration == power_iters - 1:
                diagonal_values = (basis * applied).sum(dim=1)
            basis, _ = torch.linalg.qr(applied, mode="reduced")
    elif orthogonalization in {"each_cholesky_qr", "each_cholesky_qr2"}:
        basis = previous_factor.float()
        basis = basis / torch.linalg.vector_norm(
            basis, dim=1, keepdim=True
        ).clamp_min(1e-12)
        for iteration in range(power_iters):
            applied = torch.bmm(metric_f, basis)
            if ritz_mode == "diagonal" and iteration == power_iters - 1:
                diagonal_values = (basis * applied).sum(dim=1)
            # Standard subspace iteration reorthogonalizes between power
            # applications. Cholesky-QR keeps that stabilization much cheaper
            # than a full Householder QR at these small batched ranks.
            basis, failed = cholesky_qr(
                applied,
                relative_ridge=float(cholesky_qr_ridge),
            )
            recovery_mask |= failed
            if (
                orthogonalization == "each_cholesky_qr2"
                and iteration == power_iters - 1
            ):
                # The corrective pass sees the same numerical rank as the
                # first Cholesky-QR. Reducing the ridge by 100x made the Gram
                # matrix singular for rank-64 Cosmos3 Q factors (step 5,
                # layer 4). Keep the calibrated ridge floor; this remains a
                # device-only fixed-cost operation and avoids a synchronized
                # cholesky_ex retry path.
                basis, failed = cholesky_qr(
                    basis,
                    relative_ridge=max(float(cholesky_qr_ridge), 1e-5),
                )
                recovery_mask |= failed
    else:
        # Factors produced by this module are orthogonal eigenvectors times
        # independent positive column scales. Recover the previous basis with
        # inexpensive column normalization, apply the current metric p times,
        # and orthogonalize only once at the end.
        basis = previous_factor.float()
        basis = basis / torch.linalg.vector_norm(basis, dim=1, keepdim=True).clamp_min(1e-12)
        for iteration in range(power_iters):
            applied = torch.bmm(metric_f, basis)
            if ritz_mode == "diagonal" and iteration == power_iters - 1:
                diagonal_values = (basis * applied).sum(dim=1)
            # Scaling does not change the spanned subspace and prevents the
            # two unorthogonalized power applications from overflowing.
            basis = applied / torch.linalg.vector_norm(applied, dim=1, keepdim=True).clamp_min(1e-12)
        if orthogonalization == "final_qr":
            basis, _ = torch.linalg.qr(basis, mode="reduced")
        else:
            basis, failed = cholesky_qr(basis, relative_ridge=1e-6)
            recovery_mask |= failed

    def finish(
        factor: torch.Tensor,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        nonfinite = ~torch.isfinite(factor).all(dim=(1, 2))
        nonfinite |= ~torch.isfinite(basis).all(dim=(1, 2))
        final_recovery_mask = recovery_mask | nonfinite
        if return_basis and return_recovery_mask:
            return factor, basis, final_recovery_mask
        if return_basis:
            return factor, basis
        if return_recovery_mask:
            raise ValueError("return_recovery_mask requires return_basis=True.")
        return factor

    if ritz_mode == "previous_scale":
        return finish(basis * previous_scales.unsqueeze(1))
    if ritz_mode == "diagonal":
        assert diagonal_values is not None
        eigenvalues = diagonal_values.clamp_min(0.0)
        normalizer = eigenvalues.mean(dim=1, keepdim=True).clamp_min(1e-12)
        return finish(basis * torch.sqrt(eigenvalues / normalizer).unsqueeze(1))

    rayleigh = torch.bmm(basis.transpose(1, 2), torch.bmm(metric_f, basis))
    # Remove harmless GEMM asymmetry before the reduced symmetric solve.
    rayleigh = 0.5 * (rayleigh + rayleigh.transpose(1, 2))
    if ritz_mode == "rayleigh_cholesky":
        # The full Ritz factor B U sqrt(Lambda) and B chol(R) have the same
        # Gram matrix because R = U Lambda U^T = chol(R) chol(R)^T.  Role
        # K-means depends on distances induced by F F^T, not on the arbitrary
        # right-side orientation of F, so the expensive reduced EIGH is not
        # required. A tiny relative jitter handles FP32 roundoff near zero.
        normalizer = (
            rayleigh.diagonal(dim1=1, dim2=2)
            .mean(dim=1, keepdim=True)
            .clamp_min(1e-12)
        )
        chol, failed = cholesky_with_device_fallback(
            rayleigh,
            normalizer.squeeze(1),
            relative_ridge=float(rayleigh_ridge),
            return_failure_mask=True,
        )
        recovery_mask |= failed
        return finish(torch.bmm(basis, chol) / torch.sqrt(normalizer).unsqueeze(1))
    eigenvalues, rotation = torch.linalg.eigh(rayleigh)
    eigenvalues = eigenvalues.clamp_min(0.0)
    eigenvectors = torch.bmm(basis, rotation)
    normalizer = eigenvalues.mean(dim=1, keepdim=True).clamp_min(1e-12)
    return finish(eigenvectors * torch.sqrt(eigenvalues / normalizer).unsqueeze(1))


def metric_relative_drift(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-head relative Frobenius drift, computed entirely on the device."""
    if current.shape != reference.shape:
        raise RuntimeError(
            f"Metric drift shape mismatch: current={tuple(current.shape)}, reference={tuple(reference.shape)}."
        )
    numerator = torch.linalg.vector_norm(current - reference, dim=(1, 2))
    denominator = torch.linalg.vector_norm(reference, dim=(1, 2)).clamp_min(1e-12)
    return numerator / denominator


def key_metric_from_queries(queries: torch.Tensor, sample_tokens: int) -> torch.Tensor:
    sampled = deterministic_token_sample(queries, sample_tokens)
    return covariance(sampled, center=False)


def value_metric_from_values(values: torch.Tensor, sample_tokens: int) -> torch.Tensor:
    sampled = deterministic_token_sample(values, sample_tokens)
    return covariance(sampled, center=True)


def query_factor_from_keys(keys: torch.Tensor, rank: int, sample_tokens: int) -> torch.Tensor:
    sampled = deterministic_token_sample(keys, sample_tokens)
    return metric_factor(covariance(sampled, center=True), rank)


def key_factor_from_queries(queries: torch.Tensor, rank: int, sample_tokens: int) -> torch.Tensor:
    return metric_factor(key_metric_from_queries(queries, sample_tokens), rank)


def value_factor_from_values(values: torch.Tensor, rank: int, sample_tokens: int) -> torch.Tensor:
    return metric_factor(value_metric_from_values(values, sample_tokens), rank)


def query_metric_from_key_centroids(
    centroids: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """Centered full-logit metric used to construct the Q role factor."""
    if centroids.ndim != 3 or sizes.shape != centroids.shape[:2]:
        raise RuntimeError(
            f"Centroid metric shape mismatch: centroids={tuple(centroids.shape)}, sizes={tuple(sizes.shape)}."
        )
    valid = sizes > 0
    weights = valid.float()
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (centroids.float() * weights.unsqueeze(-1)).sum(dim=1) / count.unsqueeze(-1)
    centered = (centroids.float() - mean.unsqueeze(1)) * weights.unsqueeze(-1)
    metric = torch.bmm(centered.transpose(1, 2), centered) / count.view(-1, 1, 1)
    return metric / float(centroids.shape[-1])


def query_factor_from_key_centroids(
    centroids: torch.Tensor,
    sizes: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Factor the centered full K-centroid logit metric for each KV head."""
    return metric_factor(query_metric_from_key_centroids(centroids, sizes), rank)


def project_tokens(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or factor.ndim != 3 or x.shape[0] != factor.shape[0] or x.shape[2] != factor.shape[1]:
        raise RuntimeError(f"Projection shape mismatch: x={tuple(x.shape)}, factor={tuple(factor.shape)}.")
    return torch.bmm(x.float(), factor.float())


def normalize_feature_scale(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each batched feature cloud by one RMS scale."""
    scale = features.float().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    return features.float() / scale, scale


def normalize_rows_rms(features: torch.Tensor) -> torch.Tensor:
    """Normalize projected full-logit vectors without basis-dependent centering."""
    scale = features.float().square().mean(dim=2, keepdim=True).sqrt().clamp_min(1e-6)
    return features.float() / scale


def lift_projected_centroids(centroids: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Map projected centers to minimum-norm original-space representatives."""
    if centroids.ndim != 3 or factor.ndim != 3 or centroids.shape[0] != factor.shape[0]:
        raise RuntimeError(f"Lift shape mismatch: centroids={tuple(centroids.shape)}, factor={tuple(factor.shape)}.")
    if centroids.shape[2] != factor.shape[2]:
        raise RuntimeError(f"Lift feature mismatch: centroids={tuple(centroids.shape)}, factor={tuple(factor.shape)}.")
    inverse_scales = factor.float().square().sum(dim=1).clamp_min(1e-12).reciprocal()
    pseudo_inverse = factor.float().transpose(1, 2) * inverse_scales.unsqueeze(-1)
    return torch.bmm(centroids.float(), pseudo_inverse)


def gqa_metric_sources(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_cluster_heads: int,
    k_cluster_heads: int,
    gqa_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K samples for the Q metric and Q samples for the K metric."""
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape:
        raise RuntimeError(f"Expected matching [B,H,N,D] Q/K tensors, got Q={tuple(q.shape)}, K={tuple(k.shape)}.")
    batch, heads, tokens, dim = q.shape
    if gqa_group_size <= 0:
        raise RuntimeError(f"gqa_group_size must be positive, got {gqa_group_size}.")

    if q_cluster_heads == heads:
        q_metric_keys = k.contiguous().view(batch * heads, tokens, dim)
    elif q_cluster_heads * gqa_group_size == heads:
        q_metric_keys = k[:, ::gqa_group_size].contiguous().view(batch * q_cluster_heads, tokens, dim)
    else:
        raise RuntimeError(
            f"Cannot construct Q metrics for heads={heads}, q_cluster_heads={q_cluster_heads}, G={gqa_group_size}."
        )

    if k_cluster_heads == heads:
        k_metric_queries = q.contiguous().view(batch * heads, tokens, dim)
    elif k_cluster_heads * gqa_group_size == heads:
        k_metric_queries = (
            q.contiguous()
            .view(batch, k_cluster_heads, gqa_group_size, tokens, dim)
            .view(batch * k_cluster_heads, gqa_group_size * tokens, dim)
        )
    else:
        raise RuntimeError(
            f"Cannot construct K metrics for heads={heads}, k_cluster_heads={k_cluster_heads}, G={gqa_group_size}."
        )
    return q_metric_keys, k_metric_queries


def original_centroids(
    x: torch.Tensor,
    labels: torch.Tensor,
    cluster_sizes: torch.Tensor,
    *,
    empty_fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.ndim != 3 or labels.shape != x.shape[:2]:
        raise RuntimeError(f"Centroid shape mismatch: x={tuple(x.shape)}, labels={tuple(labels.shape)}.")
    batch_heads, _, dim = x.shape
    clusters = int(cluster_sizes.shape[1])
    if cluster_sizes.shape[0] != batch_heads:
        raise RuntimeError(
            f"Cluster-size shape mismatch: sizes={tuple(cluster_sizes.shape)}, batch_heads={batch_heads}."
        )
    if _ORIGINAL_CENTROID_BACKEND == "triton_atomic_fp32" and x.is_cuda:
        from ...kernels.triton.centroid_reduction import original_centroids_triton

        return original_centroids_triton(
            x,
            labels,
            cluster_sizes,
            empty_fallback=empty_fallback,
        )
    sums = torch.zeros((batch_heads, clusters, dim), device=x.device, dtype=torch.float32)
    sums.scatter_add_(1, labels.long().unsqueeze(-1).expand(-1, -1, dim), x.float())
    means = sums / cluster_sizes.float().clamp_min(1.0).unsqueeze(-1)
    if empty_fallback is not None:
        if empty_fallback.shape != means.shape:
            raise RuntimeError(
                f"Empty-centroid fallback mismatch: fallback={tuple(empty_fallback.shape)}, means={tuple(means.shape)}."
            )
        means = torch.where(cluster_sizes.unsqueeze(-1) > 0, means, empty_fallback.float())
    return means


def joint_original_centroids(
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    cluster_sizes: torch.Tensor,
    *,
    x_empty_fallback: torch.Tensor | None = None,
    y_empty_fallback: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct two original-space centroid sets sharing one label traversal."""
    if _ORIGINAL_CENTROID_BACKEND == "triton_atomic_fp32" and x.is_cuda:
        from ...kernels.triton.centroid_reduction import joint_original_centroids_triton

        return joint_original_centroids_triton(
            x,
            y,
            labels,
            cluster_sizes,
            x_empty_fallback=x_empty_fallback,
            y_empty_fallback=y_empty_fallback,
        )
    return (
        original_centroids(
            x,
            labels,
            cluster_sizes,
            empty_fallback=x_empty_fallback,
        ),
        original_centroids(
            y,
            labels,
            cluster_sizes,
            empty_fallback=y_empty_fallback,
        ),
    )
