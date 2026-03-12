import copy
import os
from typing import Any, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _l2_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _pairwise_l2(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Compute pairwise L2 distance between a [N, F] and b [M, F] -> [N, M].
    Optimized: uses ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
    """
    a2 = jnp.sum(a * a, axis=-1, keepdims=True)  # [N,1]
    b2 = jnp.sum(b * b, axis=-1, keepdims=True)  # [M,1]
    cross = a @ b.T  # [N,M]
    dist2 = a2 + b2.T - 2.0 * cross
    dist2 = jnp.maximum(dist2, 0.0)
    return jnp.sqrt(dist2 + eps)


def _cfg_get(cfg, key, default):
    """Safe config read for FrozenDict / ConfigDict."""
    try:
        return cfg[key]
    except KeyError:
        return default


def _alg2_attention(logits: jnp.ndarray) -> jnp.ndarray:
    """
    Paper Alg.2 normalization:
      A_row = softmax(logit, dim=-1)
      A_col = softmax(logit, dim=-2)
      A = sqrt(A_row * A_col)
    logits: [N, M]
    returns: [N, M]
    """
    A_row = jax.nn.softmax(logits, axis=-1)
    A_col = jax.nn.softmax(logits, axis=-2)
    return jnp.sqrt(A_row * A_col)


def _sample_inbatch_neg_actions(actions_pool: jnp.ndarray, rng, k: int) -> jnp.ndarray:
    """
    Sample per-row K negative actions from the same minibatch.
    actions_pool: [B, A]
    returns:      [B, K, A]
    """
    B = actions_pool.shape[0]
    k = int(k)
    idx = jax.random.randint(rng, (B, k), 0, B)
    row = jnp.arange(B)[:, None]
    idx = jnp.where(idx == row, (idx + 1) % B, idx)
    return jnp.take(actions_pool, idx, axis=0)


# ============================
# local distance scale for kernel drift (legacy)
# ============================
def _kernel_local_scale(
    dist_pos: jnp.ndarray,
    dist_neg: jnp.ndarray,
    mask_self: bool,
    action_dim: int = None,
    use_sqrt_dim: bool = False,
    eps: float = 1e-12,
) -> jnp.ndarray:
    neg_mask = jnp.ones_like(dist_neg)

    if mask_self:
        nmask = min(dist_neg.shape[0], dist_neg.shape[1])
        idx = jnp.arange(nmask)
        neg_mask = neg_mask.at[idx, idx].set(0.0)

    num = jnp.sum(dist_pos) + jnp.sum(dist_neg * neg_mask)
    den = float(dist_pos.shape[0] * dist_pos.shape[1]) + jnp.sum(neg_mask)

    scale = num / (den + eps)

    if use_sqrt_dim and action_dim is not None:
        scale = scale / jnp.sqrt(jnp.asarray(action_dim, dtype=dist_pos.dtype))

    return jnp.maximum(scale, eps)


# =============================================================================
# ORIGINAL kernel drift (unchanged, for backward compatibility)
# =============================================================================
def _compute_drift_one_temp(
    gen_a: jnp.ndarray,  # [Nq, A]
    pos_a: jnp.ndarray,  # [Npos, A]
    neg_a: jnp.ndarray,  # [NnegT, A]
    mask_self: bool,     # True only if neg_a begins with gen_a in same order
    temp: float,
    w_neg: jnp.ndarray = None,     # [NnegT] optional column weights
    scale_distances: bool = False,
    affinity_norm: bool = False,
    eps: float = 1e-12,
):
    """
    ORIGINAL kernel drift (kept for backward compat / ablation):
      - Alg2 attention (row-softmax * col-softmax geometric mean)
      - Cross-weighting (W_pos = A_pos * sum(A_neg))
      - Position-based drift (W @ positions)
    """
    Nq = gen_a.shape[0]
    Npos = pos_a.shape[0]

    dist_pos = _pairwise_l2(gen_a, pos_a, eps=eps)   # [Nq, Npos]
    dist_neg = _pairwise_l2(gen_a, neg_a, eps=eps)   # [Nq, NnegT]

    if scale_distances:
        scale = _kernel_local_scale(dist_pos, dist_neg, mask_self=mask_self, eps=eps)
        dist_pos = dist_pos / scale
        dist_neg = dist_neg / scale

    # mask self-target only after scaling
    if mask_self:
        idx = jnp.arange(Nq)
        dist_neg = dist_neg.at[idx, idx].set(1e6)

    logit_pos = -dist_pos / temp
    logit_neg = -dist_neg / temp

    logits = jnp.concatenate([logit_pos, logit_neg], axis=-1)  # [Nq, Npos+NnegT]
    A = _alg2_attention(logits)

    A_pos = A[:, :Npos]   # [Nq, Npos]
    A_neg = A[:, Npos:]   # [Nq, NnegT]
    dist_all = jnp.concatenate([dist_pos, dist_neg], axis=-1)

    if affinity_norm:
        norm_est = jax.lax.stop_gradient(jnp.sqrt(jnp.mean((A * dist_all) ** 2) + eps))
    else:
        norm_est = jnp.asarray(1.0, dtype=A.dtype)

    W_pos = A_pos * jnp.sum(A_neg, axis=1, keepdims=True)  # [Nq, Npos]
    W_neg = A_neg * jnp.sum(A_pos, axis=1, keepdims=True)  # [Nq, NnegT]

    if w_neg is not None:
        W_neg = W_neg * w_neg[None, :]

    drift_pos = (W_pos @ pos_a) / norm_est  # [Nq, A]
    drift_neg = (W_neg @ neg_a) / norm_est  # [Nq, A]
    V = drift_pos - drift_neg
    return V, drift_pos, drift_neg


# =============================================================================
# V2 kernel drift: all fixes as individual toggles
# =============================================================================
def _compute_drift_one_temp_v2(
    gen_a: jnp.ndarray,       # [Nq, A]
    pos_a: jnp.ndarray,       # [Npos, A]
    neg_a: jnp.ndarray,       # [Nneg, A]
    mask_self: bool,
    temp: float,
    # --- Fix toggles ---
    use_displacement: bool = True,     # Fix 3: displacement vectors instead of positions
    row_softmax_only: bool = False,    # Fix 1: drop column-softmax (Alg2 → row-only)
    no_cross_weight: bool = False,     # Fix 2: drop cross-weighting (W = A directly)
    separate_softmax: bool = False,    # Fix 4: independent softmax for pos/neg
    dim_scale: bool = False,           # Fix 5: divide distances by sqrt(action_dim)
    beta_repel: float = 1.0,          # repulsion strength (only used with separate_softmax)
    # --- Legacy options ---
    w_neg: jnp.ndarray = None,
    scale_distances: bool = False,
    affinity_norm: bool = False,
    eps: float = 1e-12,
):
    """
    V2 kernel drift with individual fix toggles.

    Recommended progression for experiments:
      1. use_displacement=True (alone) — biggest single improvement
      2. + separate_softmax=True        — fixes Npos=1 attention dilution
      3. + dim_scale=True               — makes tau transferable across tasks
      4. + beta_repel tuning            — fine-tune attraction/repulsion balance

    When separate_softmax=True, row_softmax_only and no_cross_weight are
    automatically implied (they are sub-components of the same fix).
    """
    Nq = gen_a.shape[0]
    Npos = pos_a.shape[0]
    action_dim = gen_a.shape[-1]

    dist_pos = _pairwise_l2(gen_a, pos_a, eps=eps)   # [Nq, Npos]
    dist_neg = _pairwise_l2(gen_a, neg_a, eps=eps)    # [Nq, Nneg]

    # Legacy: local distance scaling
    if scale_distances:
        scale = _kernel_local_scale(dist_pos, dist_neg, mask_self=mask_self, eps=eps)
        dist_pos = dist_pos / scale
        dist_neg = dist_neg / scale

    # Mask self-distances
    if mask_self:
        n = min(Nq, neg_a.shape[0])
        idx = jnp.arange(n)
        dist_neg = dist_neg.at[idx, idx].set(1e6)

    # Fix 5: dimension-aware temperature scaling
    if dim_scale:
        dim_factor = jnp.sqrt(jnp.asarray(action_dim, dtype=gen_a.dtype))
    else:
        dim_factor = jnp.asarray(1.0, dtype=gen_a.dtype)

    # =====================================================================
    # BRANCH A: separate softmax (Fix 4) — implies fixes 1, 2, 3
    # This is the recommended path for Npos=1.
    # Attraction and repulsion have independent probability distributions.
    # =====================================================================
    if separate_softmax:
        logit_pos = -(dist_pos / dim_factor) / temp   # [Nq, Npos]
        logit_neg = -(dist_neg / dim_factor) / temp    # [Nq, Nneg]

        # Independent softmax: W_pos sums to 1 over positives,
        #                      W_neg sums to 1 over negatives.
        # With Npos=1: W_pos = [[1],[1],...,[1]] — full attraction for everyone.
        W_pos = jax.nn.softmax(logit_pos, axis=-1)    # [Nq, Npos]
        W_neg = jax.nn.softmax(logit_neg, axis=-1)    # [Nq, Nneg]

        if w_neg is not None:
            W_neg = W_neg * w_neg[None, :]

        # Fix 3 (always on in this branch): displacement-based drift
        disp_to_pos = pos_a[None, :, :] - gen_a[:, None, :]   # [Nq, Npos, A]
        disp_to_neg = neg_a[None, :, :] - gen_a[:, None, :]   # [Nq, Nneg, A]

        drift_pos = jnp.sum(W_pos[:, :, None] * disp_to_pos, axis=1)   # [Nq, A]
        drift_neg = jnp.sum(W_neg[:, :, None] * disp_to_neg, axis=1)   # [Nq, A]

        V = drift_pos - beta_repel * drift_neg
        return V, drift_pos, -beta_repel * drift_neg

    # =====================================================================
    # BRANCH B: combined softmax (original structure, with optional fixes)
    # Use this path for ablations or when you have multiple positives.
    # =====================================================================
    logit_pos = -(dist_pos / dim_factor) / temp
    logit_neg = -(dist_neg / dim_factor) / temp

    logits = jnp.concatenate([logit_pos, logit_neg], axis=-1)  # [Nq, Npos+Nneg]

    # Fix 1: row-softmax only vs Alg2
    if row_softmax_only:
        A = jax.nn.softmax(logits, axis=-1)
    else:
        A = _alg2_attention(logits)

    A_pos = A[:, :Npos]
    A_neg = A[:, Npos:]

    # Affinity normalization (legacy)
    if affinity_norm:
        dist_all = jnp.concatenate([dist_pos, dist_neg], axis=-1)
        norm_est = jax.lax.stop_gradient(jnp.sqrt(jnp.mean((A * dist_all) ** 2) + eps))
    else:
        norm_est = jnp.asarray(1.0, dtype=A.dtype)

    # Fix 2: cross-weighting vs direct weights
    if no_cross_weight:
        W_pos = A_pos
        W_neg = A_neg
    else:
        W_pos = A_pos * jnp.sum(A_neg, axis=1, keepdims=True)
        W_neg = A_neg * jnp.sum(A_pos, axis=1, keepdims=True)

    if w_neg is not None:
        W_neg = W_neg * w_neg[None, :]

    # Fix 3: displacement-based vs position-based
    if use_displacement:
        disp_to_pos = pos_a[None, :, :] - gen_a[:, None, :]   # [Nq, Npos, A]
        disp_to_neg = neg_a[None, :, :] - gen_a[:, None, :]   # [Nq, Nneg, A]

        drift_pos = jnp.sum(W_pos[:, :, None] * disp_to_pos, axis=1) / norm_est
        drift_neg = jnp.sum(W_neg[:, :, None] * disp_to_neg, axis=1) / norm_est
    else:
        drift_pos = (W_pos @ pos_a) / norm_est
        drift_neg = (W_neg @ neg_a) / norm_est

    V = drift_pos - drift_neg
    return V, drift_pos, drift_neg


# =============================================================================
# ORIGINAL contragen drift (unchanged)
# =============================================================================
def _compute_drift_one_temp_contragen(
    gen_a: jnp.ndarray,  # [Nq, A]
    pos_a: jnp.ndarray,  # [Npos, A]
    neg_a: jnp.ndarray,  # [NnegT, A]
    temp: float,
    include_neg: bool,
    diag_boost: float,
    force_scale: float,
    w_neg: jnp.ndarray = None,
    eps: float = 1e-12,
):
    """
    Contragen-style force field (unchanged from original).
    """
    npos = pos_a.shape[0]
    if include_neg:
        anchors = jnp.concatenate([pos_a, neg_a], axis=0)
    else:
        anchors = pos_a

    nanchor = anchors.shape[0]
    pos_neg = jnp.concatenate([anchors, gen_a], axis=0)

    dist = _pairwise_l2(pos_neg, pos_neg, eps=eps)
    dist = jax.lax.stop_gradient(dist)

    scale = jax.lax.stop_gradient(jnp.maximum(jnp.mean(dist), eps))
    scale_2 = scale / jnp.sqrt(jnp.asarray(gen_a.shape[-1], dtype=dist.dtype))
    scale_2 = jax.lax.stop_gradient(jnp.maximum(scale_2, eps))

    pos_neg_scaled = pos_neg / scale_2
    dist = dist + jnp.eye(pos_neg.shape[0], dtype=dist.dtype) * (float(diag_boost) * scale)
    dist = dist / scale

    radius = jnp.maximum(jnp.asarray(temp, dtype=dist.dtype), 1e-6)
    logits = -dist / radius
    aff_row = jax.nn.softmax(logits, axis=-1)
    affinity = jnp.sqrt(jnp.maximum(aff_row * aff_row.T, 0.0) + eps)

    if include_neg and (w_neg is not None):
        col_weights = jnp.concatenate(
            [
                jnp.ones((npos,), dtype=affinity.dtype),
                w_neg.astype(affinity.dtype),
                jnp.ones((gen_a.shape[0],), dtype=affinity.dtype),
            ],
            axis=0,
        )
        affinity = affinity * col_weights[None, :]

    norm_est = jax.lax.stop_gradient(jnp.sqrt(jnp.mean((affinity * dist) ** 2) + eps))

    aff_samp_anchor = affinity[nanchor:, :nanchor]
    aff_samp_samp = affinity[nanchor:, nanchor:]

    coeff_anchor = aff_samp_anchor * jnp.sum(aff_samp_samp, axis=-1, keepdims=True)
    coeff_samp = -aff_samp_samp * jnp.sum(aff_samp_anchor, axis=-1, keepdims=True)
    coeff = jnp.concatenate([coeff_anchor, coeff_samp], axis=-1) / norm_est

    force_scaled = coeff @ pos_neg_scaled
    V_s = force_scaled * float(force_scale)

    coeff_pos = jnp.maximum(coeff, 0.0)
    coeff_neg = jnp.maximum(-coeff, 0.0)
    drift_pos_scaled = coeff_pos @ pos_neg_scaled
    drift_neg_scaled = coeff_neg @ pos_neg_scaled
    drift_pos_s = drift_pos_scaled * float(force_scale)
    drift_neg_s = drift_neg_scaled * float(force_scale)
    return V_s, drift_pos_s, drift_neg_s, scale_2


# =============================================================================
# Main dispatch: compute_drift_field_conditional
# =============================================================================
def compute_drift_field_conditional(
    gen_a: jnp.ndarray,  # [B, Nq, A]
    pos_a: jnp.ndarray,  # [B, Npos, A]
    neg_a: jnp.ndarray,  # [B, NnegT, A]
    mask_self: bool,
    temps: Sequence[float],
    drift_normalize: bool,
    drift_style: str = "kernel",           # "kernel" | "kernel_v2" | "contragen"
    drift_contragen_use_negatives: bool = False,
    contra_diag_boost: float = 100.0,
    contra_force_scale: float = 1.0,
    w_neg: jnp.ndarray = None,
    # --- Legacy kernel flags ---
    scale_distances: bool = False,
    global_rms_normalize: bool = False,
    kernel_affinity_norm: bool = False,
    # --- V2 kernel fix toggles ---
    v2_use_displacement: bool = True,
    v2_row_softmax_only: bool = False,
    v2_no_cross_weight: bool = False,
    v2_separate_softmax: bool = False,
    v2_dim_scale: bool = False,
    v2_beta_repel: float = 1.0,
    eps: float = 1e-12,
):
    """
    Returns:
      V_total, drift_pos_total, drift_neg_total,
      v_norms, pos_norms, neg_norms,
      v_sqs, pos_sqs, neg_sqs,
      lams, v_raw_norms, v_raw_sqs
    """

    # -----------------------------------------------------------------
    # CONTRAGEN BRANCH (unchanged)
    # -----------------------------------------------------------------
    if drift_style == "contragen":
        contragen_eps = max(float(eps), 1e-8)
        v_s_list, dp_s_list, dn_s_list = [], [], []
        scale2 = None

        for T in temps:
            if w_neg is None:
                v_s_t, dp_s_t, dn_s_t, scale2_t = jax.vmap(
                    lambda g, p, n: _compute_drift_one_temp_contragen(
                        gen_a=g, pos_a=p, neg_a=n,
                        temp=float(T),
                        include_neg=drift_contragen_use_negatives,
                        diag_boost=float(contra_diag_boost),
                        force_scale=float(contra_force_scale),
                        w_neg=None, eps=contragen_eps,
                    ),
                    in_axes=(0, 0, 0),
                )(gen_a, pos_a, neg_a)
            else:
                v_s_t, dp_s_t, dn_s_t, scale2_t = jax.vmap(
                    lambda g, p, n, w: _compute_drift_one_temp_contragen(
                        gen_a=g, pos_a=p, neg_a=n,
                        temp=float(T),
                        include_neg=drift_contragen_use_negatives,
                        diag_boost=float(contra_diag_boost),
                        force_scale=float(contra_force_scale),
                        w_neg=w, eps=contragen_eps,
                    ),
                    in_axes=(0, 0, 0, 0),
                )(gen_a, pos_a, neg_a, w_neg)
            v_s_list.append(v_s_t)
            dp_s_list.append(dp_s_t)
            dn_s_list.append(dn_s_t)
            scale2 = scale2_t

        v_s_total = jnp.sum(jnp.stack(v_s_list, axis=0), axis=0)
        dp_s_total = jnp.sum(jnp.stack(dp_s_list, axis=0), axis=0)
        dn_s_total = jnp.sum(jnp.stack(dn_s_list, axis=0), axis=0)
        batch_force_rms = jax.lax.stop_gradient(
            jnp.sqrt(jnp.mean(v_s_total * v_s_total) + contragen_eps)
        )

        v_s_total = v_s_total / batch_force_rms
        dp_s_total = dp_s_total / batch_force_rms
        dn_s_total = dn_s_total / batch_force_rms

        V_sum = v_s_total * scale2[:, None, None]
        pos_sum = dp_s_total * scale2[:, None, None]
        neg_sum = dn_s_total * scale2[:, None, None]

        act_dim = gen_a.shape[-1]
        v_norm_list, pos_norm_list, neg_norm_list = [], [], []
        v_sq_list, pos_sq_list, neg_sq_list = [], [], []
        lam_list, v_raw_norm_list, v_raw_sq_list = [], [], []

        for v_s_t, dp_s_t, dn_s_t in zip(v_s_list, dp_s_list, dn_s_list):
            v_t = (v_s_t / batch_force_rms) * scale2[:, None, None]
            dp_t = (dp_s_t / batch_force_rms) * scale2[:, None, None]
            dn_t = (dn_s_t / batch_force_rms) * scale2[:, None, None]

            raw_sq = jnp.mean(jnp.sum(v_t * v_t, axis=-1), axis=-1) / act_dim
            raw_norm = jnp.mean(_l2_norm(v_t, axis=-1), axis=-1)
            lam = jax.lax.stop_gradient(jnp.sqrt(raw_sq + contragen_eps))

            v_norm_list.append(jnp.mean(_l2_norm(v_t, axis=-1), axis=-1))
            pos_norm_list.append(jnp.mean(_l2_norm(dp_t, axis=-1), axis=-1))
            neg_norm_list.append(jnp.mean(_l2_norm(dn_t, axis=-1), axis=-1))
            v_sq_list.append(jnp.mean(jnp.sum(v_t * v_t, axis=-1), axis=-1))
            pos_sq_list.append(jnp.mean(jnp.sum(dp_t * dp_t, axis=-1), axis=-1))
            neg_sq_list.append(jnp.mean(jnp.sum(dn_t * dn_t, axis=-1), axis=-1))
            lam_list.append(lam)
            v_raw_norm_list.append(raw_norm)
            v_raw_sq_list.append(raw_sq)

        return (
            V_sum, pos_sum, neg_sum,
            jnp.stack(v_norm_list, axis=1),
            jnp.stack(pos_norm_list, axis=1),
            jnp.stack(neg_norm_list, axis=1),
            jnp.stack(v_sq_list, axis=1),
            jnp.stack(pos_sq_list, axis=1),
            jnp.stack(neg_sq_list, axis=1),
            jnp.stack(lam_list, axis=1),
            jnp.stack(v_raw_norm_list, axis=1),
            jnp.stack(v_raw_sq_list, axis=1),
        )

    # -----------------------------------------------------------------
    # KERNEL BRANCHES (original or v2)
    # -----------------------------------------------------------------
    use_v2 = (drift_style == "kernel_v2")

    def per_item(gen_a_i, pos_a_i, neg_a_i, w_i):
        V_sum = jnp.zeros_like(gen_a_i)
        pos_sum = jnp.zeros_like(gen_a_i)
        neg_sum = jnp.zeros_like(gen_a_i)

        v_norm_list, pos_norm_list, neg_norm_list = [], [], []
        v_sq_list, pos_sq_list, neg_sq_list = [], [], []
        lam_list, v_raw_norm_list, v_raw_sq_list = [], [], []

        for T in temps:
            if use_v2:
                V_T, dp_T, dn_T = _compute_drift_one_temp_v2(
                    gen_a=gen_a_i,
                    pos_a=pos_a_i,
                    neg_a=neg_a_i,
                    mask_self=mask_self,
                    temp=float(T),
                    # --- Fix toggles ---
                    use_displacement=v2_use_displacement,
                    row_softmax_only=v2_row_softmax_only,
                    no_cross_weight=v2_no_cross_weight,
                    separate_softmax=v2_separate_softmax,
                    dim_scale=v2_dim_scale,
                    beta_repel=float(v2_beta_repel),
                    # --- Legacy ---
                    w_neg=w_i,
                    scale_distances=scale_distances,
                    affinity_norm=kernel_affinity_norm,
                    eps=eps,
                )
            else:
                V_T, dp_T, dn_T = _compute_drift_one_temp(
                    gen_a=gen_a_i,
                    pos_a=pos_a_i,
                    neg_a=neg_a_i,
                    mask_self=mask_self,
                    temp=float(T),
                    w_neg=w_i,
                    scale_distances=scale_distances,
                    affinity_norm=kernel_affinity_norm,
                    eps=eps,
                )

            act_dim = gen_a_i.shape[-1]
            raw_sq = jnp.mean(jnp.sum(V_T * V_T, axis=-1)) / act_dim
            raw_norm = jnp.mean(_l2_norm(V_T, axis=-1))
            lam = jax.lax.stop_gradient(jnp.sqrt(raw_sq + eps))

            lam_list.append(lam)
            v_raw_norm_list.append(raw_norm)
            v_raw_sq_list.append(raw_sq)

            if drift_normalize:
                V_T = V_T / lam
                dp_T = dp_T / lam
                dn_T = dn_T / lam

            V_sum = V_sum + V_T
            pos_sum = pos_sum + dp_T
            neg_sum = neg_sum + dn_T

            v_norm_list.append(jnp.mean(_l2_norm(V_T, axis=-1)))
            pos_norm_list.append(jnp.mean(_l2_norm(dp_T, axis=-1)))
            neg_norm_list.append(jnp.mean(_l2_norm(dn_T, axis=-1)))

            v_sq_list.append(jnp.mean(jnp.sum(V_T * V_T, axis=-1)))
            pos_sq_list.append(jnp.mean(jnp.sum(dp_T * dp_T, axis=-1)))
            neg_sq_list.append(jnp.mean(jnp.sum(dn_T * dn_T, axis=-1)))

        return (
            V_sum, pos_sum, neg_sum,
            jnp.stack(v_norm_list, axis=0),
            jnp.stack(pos_norm_list, axis=0),
            jnp.stack(neg_norm_list, axis=0),
            jnp.stack(v_sq_list, axis=0),
            jnp.stack(pos_sq_list, axis=0),
            jnp.stack(neg_sq_list, axis=0),
            jnp.stack(lam_list, axis=0),
            jnp.stack(v_raw_norm_list, axis=0),
            jnp.stack(v_raw_sq_list, axis=0),
        )

    if w_neg is None:
        outs = jax.vmap(per_item, in_axes=(0, 0, 0, None))(gen_a, pos_a, neg_a, None)
    else:
        outs = jax.vmap(per_item, in_axes=(0, 0, 0, 0))(gen_a, pos_a, neg_a, w_neg)

    V_sum, pos_sum, neg_sum = outs[0], outs[1], outs[2]
    rest = outs[3:]

    if global_rms_normalize:
        global_rms = jax.lax.stop_gradient(jnp.sqrt(jnp.mean(V_sum * V_sum) + eps))
        V_sum = V_sum / global_rms
        pos_sum = pos_sum / global_rms
        neg_sum = neg_sum / global_rms

    return (V_sum, pos_sum, neg_sum, *rest)


# =============================================================================
# Agent
# =============================================================================
class DriftQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch["next_observations"], seed=sample_rng)
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], actions=next_actions
        )

        if self.config["q_agg"] == "min":
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
        q = self.network.select("critic")(
            batch["observations"], actions=batch["actions"], params=grad_params
        )
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def drifting_bc_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch["actions"].shape
        rng, noise_rng, sub_rng, neg_rng = jax.random.split(rng, 4)

        drift_style = str(_cfg_get(self.config, "drift_style", "kernel"))
        drift_contragen_use_negatives = bool(
            _cfg_get(self.config, "drift_contragen_use_negatives", False)
        )
        contra_diag_boost = float(_cfg_get(self.config, "contra_diag_boost", 100.0))
        contra_force_scale = float(_cfg_get(self.config, "contra_force_scale", 1.0))
        is_contragen = drift_style == "contragen"
        is_v2 = drift_style == "kernel_v2"

        # Legacy kernel flags
        drift_scale_distances = bool(_cfg_get(self.config, "drift_scale_distances", False))
        drift_global_rms_normalize = bool(
            _cfg_get(self.config, "drift_global_rms_normalize", False)
        )
        drift_keep_gen_repel = bool(_cfg_get(self.config, "drift_keep_gen_repel", False))
        drift_kernel_affinity_norm = bool(
            _cfg_get(self.config, "drift_kernel_affinity_norm", False)
        )

        # V2 fix toggles
        v2_use_displacement = bool(_cfg_get(self.config, "v2_use_displacement", True))
        v2_row_softmax_only = bool(_cfg_get(self.config, "v2_row_softmax_only", False))
        v2_no_cross_weight = bool(_cfg_get(self.config, "v2_no_cross_weight", False))
        v2_separate_softmax = bool(_cfg_get(self.config, "v2_separate_softmax", False))
        v2_dim_scale = bool(_cfg_get(self.config, "v2_dim_scale", False))
        v2_beta_repel = float(_cfg_get(self.config, "v2_beta_repel", 1.0))

        # Optional subsample
        drift_bs = int(self.config["drift_batch_size"])
        if drift_bs < batch_size:
            idx = jax.random.choice(sub_rng, batch_size, (drift_bs,), replace=False)
            obs = batch["observations"][idx]
            pos_actions_pool = batch["actions"][idx]
        else:
            obs = batch["observations"]
            pos_actions_pool = batch["actions"]
            drift_bs = batch_size

        pos_actions = pos_actions_pool[:, None, :]  # [B,1,A]
        if not is_contragen:
            pos_actions = jnp.clip(pos_actions, -1.0, 1.0)

        # Generated samples
        Nneg = int(self.config["drift_nneg"])
        noises = jax.random.normal(noise_rng, (drift_bs * Nneg, action_dim))
        obs_rep = jnp.repeat(obs, repeats=Nneg, axis=0)
        bc_raw = self.network.select("actor_bc_drift")(obs_rep, noises, params=grad_params)
        bc_actions = bc_raw.reshape(drift_bs, Nneg, action_dim)
        if not is_contragen:
            bc_actions = jnp.clip(bc_actions, -1.0, 1.0)

        neg_mode = self.config["drift_neg_mode"]
        nneg_data = int(self.config["drift_nneg_data"])
        data_scale = float(self.config["drift_neg_data_scale"])
        gen_scale = float(self.config["drift_neg_gen_scale"])

        data_negs = None
        if neg_mode in ("data", "mix"):
            data_negs = _sample_inbatch_neg_actions(pos_actions_pool, neg_rng, nneg_data)
            if not is_contragen:
                data_negs = jnp.clip(data_negs, -1.0, 1.0)

        # Build negative target set
        if neg_mode == "gen":
            neg_a = bc_actions
            mask_self = True
            w_neg = None

        elif neg_mode == "data":
            if drift_keep_gen_repel:
                neg_a = jnp.concatenate([bc_actions, data_negs], axis=1)
                mask_self = True
                w_neg = jnp.concatenate(
                    [
                        jnp.full((drift_bs, bc_actions.shape[1]), gen_scale),
                        jnp.full((drift_bs, nneg_data), data_scale),
                    ],
                    axis=1,
                )
            else:
                neg_a = data_negs
                mask_self = False
                w_neg = jnp.full((drift_bs, nneg_data), data_scale)

        elif neg_mode == "mix":
            neg_a = jnp.concatenate([bc_actions, data_negs], axis=1)
            mask_self = True
            w_neg = jnp.concatenate(
                [
                    jnp.full((drift_bs, bc_actions.shape[1]), gen_scale),
                    jnp.full((drift_bs, nneg_data), data_scale),
                ],
                axis=1,
            )
        else:
            raise ValueError(f"Unknown drift_neg_mode={neg_mode}")

        temps_cfg = self.config["drift_temps"]
        temps = (float(temps_cfg),) if isinstance(temps_cfg, (int, float)) else tuple(temps_cfg)

        if is_contragen:
            gen_for_drift = jax.lax.stop_gradient(bc_actions)
            pos_for_drift = jax.lax.stop_gradient(pos_actions)
        else:
            gen_for_drift = bc_actions
            pos_for_drift = pos_actions

        (
            V,
            drift_pos_total,
            drift_neg_total,
            v_norms,
            pos_norms,
            neg_norms,
            v_sqs,
            pos_sqs,
            neg_sqs,
            lams,
            v_raw_norms,
            v_raw_sqs,
        ) = compute_drift_field_conditional(
            gen_a=gen_for_drift,
            pos_a=pos_for_drift,
            neg_a=neg_a,
            mask_self=mask_self,
            temps=temps,
            drift_normalize=bool(self.config["drift_normalize"]),
            drift_style=drift_style,
            drift_contragen_use_negatives=drift_contragen_use_negatives,
            contra_diag_boost=contra_diag_boost,
            contra_force_scale=contra_force_scale,
            w_neg=w_neg,
            # Legacy kernel
            scale_distances=drift_scale_distances,
            global_rms_normalize=drift_global_rms_normalize,
            kernel_affinity_norm=drift_kernel_affinity_norm,
            # V2 fixes
            v2_use_displacement=v2_use_displacement,
            v2_row_softmax_only=v2_row_softmax_only,
            v2_no_cross_weight=v2_no_cross_weight,
            v2_separate_softmax=v2_separate_softmax,
            v2_dim_scale=v2_dim_scale,
            v2_beta_repel=v2_beta_repel,
            eps=float(self.config["drift_eps"]),
        )

        eta = float(self.config["drift_eta"])
        if is_contragen:
            contragen_eps = max(float(self.config["drift_eps"]), 1e-8)
            if drift_contragen_use_negatives:
                anchors_for_scale = jnp.concatenate([pos_actions, neg_a], axis=1)
            else:
                anchors_for_scale = pos_actions
            pos_neg_for_scale = jnp.concatenate([anchors_for_scale, bc_actions], axis=1)
            pos_neg_for_scale = jax.lax.stop_gradient(pos_neg_for_scale)
            scale = jax.vmap(
                lambda x: jnp.mean(_pairwise_l2(x, x, eps=contragen_eps))
            )(pos_neg_for_scale)
            scale = jnp.maximum(scale[:, None, None], contragen_eps)
            scale_2 = scale / jnp.sqrt(jnp.asarray(action_dim, dtype=bc_actions.dtype))
            scale_2 = jnp.maximum(scale_2, contragen_eps)

            recon = bc_actions / scale_2
            force = V / scale_2
            target = jax.lax.stop_gradient(recon + force)
            bc_drift_loss = jnp.mean((recon - target) ** 2)
        else:
            target = jax.lax.stop_gradient(jnp.clip(bc_actions + eta * V, -1.0, 1.0))
            bc_drift_loss = jnp.mean((bc_actions - target) ** 2)

        # --- diagnostics ---
        if is_contragen:
            x = jnp.clip(bc_actions, -1.0, 1.0)
            y = jnp.clip(pos_actions, -1.0, 1.0)
            x_drift_preclip = bc_actions + V
        else:
            x = bc_actions
            y = pos_actions
            x_drift_preclip = x + eta * V

        x_drift = jnp.clip(x_drift_preclip, -1.0, 1.0)

        drift_clip_frac = jnp.mean((jnp.abs(x_drift_preclip) > 1.0).astype(jnp.float32))
        bc_oob_frac = jnp.mean((jnp.abs(bc_raw) > 1.0).astype(jnp.float32))

        per_sample_pre = jnp.mean((x - y) ** 2, axis=-1)
        per_sample_post = jnp.mean((x_drift - y) ** 2, axis=-1)

        gt_mse_pre = jnp.mean(per_sample_pre)
        gt_mse_post = jnp.mean(per_sample_post)
        gt_mse_improve = gt_mse_pre - gt_mse_post

        gt_mse_min_pre = jnp.mean(jnp.min(per_sample_pre, axis=1))
        gt_mse_min_post = jnp.mean(jnp.min(per_sample_post, axis=1))
        gt_mse_min_improve = gt_mse_min_pre - gt_mse_min_post

        Nneg_local = per_sample_pre.shape[1]
        k = jnp.maximum(1, (Nneg_local * 10) // 100)
        sorted_pre = jnp.sort(per_sample_pre, axis=1)
        sorted_post = jnp.sort(per_sample_post, axis=1)
        gt_mse_p10_pre = jnp.mean(sorted_pre[:, k - 1])
        gt_mse_p10_post = jnp.mean(sorted_post[:, k - 1])
        gt_mse_p10_improve = gt_mse_p10_pre - gt_mse_p10_post

        p10_thresh = sorted_pre[:, k - 1][:, None]
        mask = (per_sample_pre <= p10_thresh).astype(jnp.float32)

        dir_to_gt = y - x
        v_norm = _l2_norm(V, axis=-1, eps=1e-12)
        gt_norm = _l2_norm(dir_to_gt, axis=-1, eps=1e-12)
        cos_per = jnp.sum(V * dir_to_gt, axis=-1) / (v_norm * gt_norm + 1e-12)
        cos_align = jnp.mean(cos_per)
        cos_align_all = jnp.mean(cos_per)
        cos_align_p10 = jnp.sum(cos_per * mask) / (jnp.sum(mask) + 1e-12)

        lam_mean = jnp.mean(lams)
        lam_min = jnp.min(lams)
        lam_max = jnp.max(lams)

        v_raw_sq_mean = jnp.mean(v_raw_sqs)
        v_raw_norm_mean = jnp.mean(v_raw_norms)

        drift_norm_total = jnp.mean(_l2_norm(V, axis=-1))
        drift_sq_total = jnp.mean(jnp.sum(V * V, axis=-1))
        drift_pos_norm_total = jnp.mean(_l2_norm(drift_pos_total, axis=-1))
        drift_neg_norm_total = jnp.mean(_l2_norm(drift_neg_total, axis=-1))
        drift_pos_sq_total = jnp.mean(jnp.sum(drift_pos_total * drift_pos_total, axis=-1))
        drift_neg_sq_total = jnp.mean(jnp.sum(drift_neg_total * drift_neg_total, axis=-1))

        info = {
            "bc_drift_loss": bc_drift_loss,
            "drift_norm": drift_norm_total,
            "drift_sq_total": drift_sq_total,
            "drift_pos_norm_total": drift_pos_norm_total,
            "drift_neg_norm_total": drift_neg_norm_total,
            "drift_pos_sq_total": drift_pos_sq_total,
            "drift_neg_sq_total": drift_neg_sq_total,
            "npos": jnp.array(pos_actions.shape[1]),
            "nneg": jnp.array(bc_actions.shape[1]),
            "drift/is_contragen": jnp.array(1 if is_contragen else 0),
            "drift/is_v2": jnp.array(1 if is_v2 else 0),
            "drift/contragen_use_neg": jnp.array(1 if drift_contragen_use_negatives else 0),
            "drift/scale_distances": jnp.array(1 if drift_scale_distances else 0),
            "drift/global_rms_normalize": jnp.array(1 if drift_global_rms_normalize else 0),
            "drift/keep_gen_repel": jnp.array(1 if drift_keep_gen_repel else 0),
            "drift/kernel_affinity_norm": jnp.array(1 if drift_kernel_affinity_norm else 0),
            # V2 flags
            "drift/v2_use_displacement": jnp.array(1 if v2_use_displacement else 0),
            "drift/v2_row_softmax_only": jnp.array(1 if v2_row_softmax_only else 0),
            "drift/v2_no_cross_weight": jnp.array(1 if v2_no_cross_weight else 0),
            "drift/v2_separate_softmax": jnp.array(1 if v2_separate_softmax else 0),
            "drift/v2_dim_scale": jnp.array(1 if v2_dim_scale else 0),
            "drift/v2_beta_repel": jnp.array(v2_beta_repel),
            # Quality metrics
            "drift/gt_mse_pre": gt_mse_pre,
            "drift/cos_align_all": cos_align_all,
            "drift/cos_align_p10": cos_align_p10,
            "drift/gt_mse_p10_pre": gt_mse_p10_pre,
            "drift/gt_mse_p10_post": gt_mse_p10_post,
            "drift/gt_mse_p10_improve": gt_mse_p10_improve,
            "drift/gt_mse_min_pre": gt_mse_min_pre,
            "drift/gt_mse_min_post": gt_mse_min_post,
            "drift/gt_mse_min_improve": gt_mse_min_improve,
            "drift/cos_align": cos_align,
            "drift/drift_clip_frac": drift_clip_frac,
            "drift/bc_oob_frac": bc_oob_frac,
            "drift/lam_mean": lam_mean,
            "drift/lam_min": lam_min,
            "drift/lam_max": lam_max,
            "drift/v_raw_sq_mean": v_raw_sq_mean,
            "drift/v_raw_norm_mean": v_raw_norm_mean,
        }

        for i, T in enumerate(temps):
            tau_key = f"{float(T):.4f}".rstrip("0").rstrip(".").replace(".", "p")
            info[f"drift/tau_{tau_key}/norm"] = jnp.mean(v_norms[:, i])
            info[f"drift/tau_{tau_key}/sq"] = jnp.mean(v_sqs[:, i])
            info[f"drift/tau_{tau_key}/pos_norm"] = jnp.mean(pos_norms[:, i])
            info[f"drift/tau_{tau_key}/neg_norm"] = jnp.mean(neg_norms[:, i])
            info[f"drift/tau_{tau_key}/pos_sq"] = jnp.mean(pos_sqs[:, i])
            info[f"drift/tau_{tau_key}/neg_sq"] = jnp.mean(neg_sqs[:, i])
            info[f"drift/tau_{tau_key}/lam"] = jnp.mean(lams[:, i])
            info[f"drift/tau_{tau_key}/raw_norm"] = jnp.mean(v_raw_norms[:, i])
            info[f"drift/tau_{tau_key}/raw_sq"] = jnp.mean(v_raw_sqs[:, i])

        return bc_drift_loss, info

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch["actions"].shape
        rng, noise_rng, bc_rng = jax.random.split(rng, 3)

        bc_drift_loss, bc_info = self.drifting_bc_loss(batch, grad_params, bc_rng)

        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        actor_raw = self.network.select("actor_bc_drift")(
            batch["observations"], noises, params=grad_params
        )
        actor_raw = jnp.clip(actor_raw, -1.0, 1.0)
        qs = self.network.select("critic")(batch["observations"], actions=actor_raw)

        qagg_actor = _cfg_get(self.config, "q_agg_actor", None)
        if qagg_actor is None:
            qagg_actor = self.config["q_agg"]
        if qagg_actor == "min":
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

        q_loss = -q.mean()

        if self.config["normalize_q_loss"]:
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
            q_loss = lam * q_loss

        actor_loss = self.config["alpha"] * bc_drift_loss + q_loss

        actions_for_mse = self.sample_actions(batch["observations"], seed=rng)
        mse = jnp.mean((actions_for_mse - batch["actions"]) ** 2)

        info = {
            "actor_loss": actor_loss,
            "bc_drift_loss": bc_info["bc_drift_loss"] * self.config["alpha"],
            "bc_drift_loss_raw": bc_info["bc_drift_loss"],
            "drift_norm": bc_info["drift_norm"],
            "q_loss": q_loss,
            "q": q.mean(),
            "mse": mse,
            "npos": bc_info["npos"],
            "nneg": bc_info["nneg"],
        }

        for k, v in bc_info.items():
            if k not in info:
                info[k] = v

        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["action_dim"],
            ),
        )
        raw_actions = self.network.select("actor_bc_drift")(observations, noises)
        actions = jnp.clip(raw_actions, -1.0, 1.0)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_bc_drift"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )

        actor_bc_drift_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_bc_drift"),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_drift=(actor_bc_drift_def, (ex_observations, ex_actions)),
        )

        if encoders.get("actor_bc_drift") is not None:
            network_info["actor_bc_drift_encoder"] = (
                encoders.get("actor_bc_drift"),
                (ex_observations,),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# =============================================================================
# Config
# =============================================================================
def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="driftql",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg="min",
            q_agg_actor="mean",
            alpha=10.0,
            drift_eps=1e-12,
            drift_nneg=32,
            drift_npos=1,
            drift_temps=(0.2,),  # FIXED: trailing comma to make it a tuple

            # -------------------------------------------------------
            # drift style: "kernel" (original), "kernel_v2" (fixed), "contragen"
            # -------------------------------------------------------
            drift_style="kernel_v2",

            # contragen-specific (unchanged)
            contra_diag_boost=100.0,
            contra_force_scale=1.0,
            drift_contragen_use_negatives=False,

            # legacy kernel flags (used by drift_style="kernel")
            drift_scale_distances=False,
            drift_global_rms_normalize=False,
            drift_keep_gen_repel=False,
            drift_kernel_affinity_norm=False,

            # ------------------------------------------------
            # V2 FIX TOGGLES (used by drift_style="kernel_v2")
            # ------------------------------------------------
            # Recommended experiment order:
            # ------------------------------------------------
            #   Experiment 1 — displacement only (safest, biggest win):
            #     v2_use_displacement=True
            #     (all others False/default)
            # -------------------------------------------------v
            #   Experiment 2 — displacement + separate softmax:
            #     v2_use_displacement=True
            #     v2_separate_softmax=True
            # -------------------------------------------------
            #   Experiment 3 — full fix stack:
            #     v2_use_displacement=True
            #     v2_separate_softmax=True
            #     v2_dim_scale=True
            # -------------------------------------------------
            #   Experiment 4 — full fix + repulsion tuning:
            #     v2_use_displacement=True
            #     v2_separate_softmax=True
            #     v2_dim_scale=True
            #     v2_beta_repel=0.5  (try 0.3, 0.5, 1.0, 2.0)
            #
            # For ablation, you can also try the intermediate fixes:
            #   - v2_row_softmax_only=True (without separate_softmax)
            #   - v2_no_cross_weight=True  (without separate_softmax)
            # These are subsumed by separate_softmax=True but useful
            # for understanding which component matters.
            # -------------------------------------------------------
            v2_use_displacement=True,       # Fix 3: displacement vectors, not positions
            v2_row_softmax_only=False,      # Fix 1: drop column-softmax
            v2_no_cross_weight=False,       # Fix 2: drop cross-weighting
            v2_separate_softmax=True,      # Fix 4: independent pos/neg softmax
            v2_dim_scale=False,             # Fix 5: tau / sqrt(action_dim)
            v2_beta_repel=1.0,              # repulsion strength multiplier

            # shared
            drift_normalize=True,
            drift_eta=1.0,
            drift_batch_size=256,
            normalize_q_loss=False,

            # negative source control
            drift_neg_mode="gen",
            drift_nneg_data=32,
            drift_neg_data_scale=1.0,
            drift_neg_gen_scale=1.0,

            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
