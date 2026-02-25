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


def _cfg_get(cfg, key, default):
    """Safe config read for FrozenDict / ConfigDict."""
    try:
        return cfg[key]
    except KeyError:
        return default


def _l2_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _pairwise_l2(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Compute pairwise L2 distance between a [N, F] and b [M, F] -> [N, M].

    Optimized: uses ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
    (avoids materializing [N,M,F]).
    """
    a2 = jnp.sum(a * a, axis=-1, keepdims=True)  # [N,1]
    b2 = jnp.sum(b * b, axis=-1, keepdims=True)  # [M,1]
    cross = a @ b.T  # [N,M]
    dist2 = a2 + b2.T - 2.0 * cross  # [N,M]
    dist2 = jnp.maximum(dist2, 0.0)  # numerical safety
    return jnp.sqrt(dist2 + eps)


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


def _compute_drift_one_temp(
    gen_a: jnp.ndarray,  # [Nneg, A]
    pos_a: jnp.ndarray,  # [Npos, A]
    temp: float,
    w_neg: jnp.ndarray = None,  # [Nneg] optional: Q-based "badness" for repulsion targets
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Compute drift V for ONE conditioning state (one group),
    using Alg.2 normalization, returning drift vectors in ACTION space.

    Drift uses:
      V = drift_pos - drift_neg
    where drift_pos is weighted sum of pos actions, drift_neg is weighted sum of neg actions.

    If w_neg is provided: repulsion is downweighted for high-value ("not bad") targets.
    """
    Nneg, _act_dim = gen_a.shape
    Npos = pos_a.shape[0]  # should be 1

    # Distances in action space only
    dist_pos = _pairwise_l2(gen_a, pos_a, eps=eps)  # [Nneg, 1]
    dist_neg = _pairwise_l2(gen_a, gen_a, eps=eps)  # [Nneg, Nneg]

    # mask self-negatives
    idx = jnp.arange(Nneg)
    dist_neg = dist_neg.at[idx, idx].set(1e6)

    # logits
    logit_pos = -dist_pos / temp
    logit_neg = -dist_neg / temp

    logits = jnp.concatenate([logit_pos, logit_neg], axis=-1)  # [Nneg, 1+Nneg]
    A = _alg2_attention(logits)

    A_pos = A[:, :Npos]  # [Nneg, 1]
    A_neg = A[:, Npos:]  # [Nneg, Nneg]

    W_pos = A_pos * jnp.sum(A_neg, axis=1, keepdims=True)  # [Nneg, 1]
    W_neg = A_neg * jnp.sum(A_pos, axis=1, keepdims=True)  # [Nneg, Nneg]

    # ---- Q-weighted repulsion (Iteration 4) ----
    # w_neg[k] ~= 1 => treat target k as "bad" and repel from it
    # w_neg[k] ~= 0 => do NOT repel from target k (could be unknown-but-good)
    if w_neg is not None:
        W_neg = W_neg * w_neg[None, :]  # scale columns (targets)

    drift_pos = W_pos @ pos_a  # [Nneg, A]
    drift_neg = W_neg @ gen_a  # [Nneg, A]
    V = drift_pos - drift_neg

    return V, drift_pos, drift_neg


def compute_drift_field_conditional(
    gen_a: jnp.ndarray,  # [B, Nneg, A]
    pos_a: jnp.ndarray,  # [B, Npos, A]
    temps: Sequence[float],
    drift_normalize: bool,
    w_neg: jnp.ndarray = None,  # [B, Nneg] optional
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Batched conditional drift with extra returns for logging.

    Returns:
      V_total:           [B, Nneg, A]
      drift_pos_total:   [B, Nneg, A]
      drift_neg_total:   [B, Nneg, A]
      v_norms:           [B, T]
      pos_norms:         [B, T]
      neg_norms:         [B, T]
      v_sqs:             [B, T]
      pos_sqs:           [B, T]
      neg_sqs:           [B, T]
      lams:              [B, T]
      v_raw_norms:       [B, T]
      v_raw_sqs:         [B, T]
    """

    def per_item(gen_a_i, pos_a_i, w_i):
        V_sum = jnp.zeros_like(gen_a_i)
        pos_sum = jnp.zeros_like(gen_a_i)
        neg_sum = jnp.zeros_like(gen_a_i)

        v_norm_list = []
        pos_norm_list = []
        neg_norm_list = []
        v_sq_list = []
        pos_sq_list = []
        neg_sq_list = []
        lam_list = []
        v_raw_norm_list = []
        v_raw_sq_list = []

        for T in temps:
            V_T, dp_T, dn_T = _compute_drift_one_temp(
                gen_a=gen_a_i,
                pos_a=pos_a_i,
                temp=float(T),
                w_neg=w_i,
                eps=eps,
            )

            act_dim = gen_a_i.shape[-1]
            raw_sq = jnp.mean(jnp.sum(V_T * V_T, axis=-1)) / act_dim
            raw_norm = jnp.mean(_l2_norm(V_T, axis=-1))
            lam = jnp.sqrt(raw_sq + eps)
            lam = jax.lax.stop_gradient(lam)

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
            V_sum,
            pos_sum,
            neg_sum,
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
        return jax.vmap(per_item, in_axes=(0, 0, None))(gen_a, pos_a, None)
    return jax.vmap(per_item, in_axes=(0, 0, 0))(gen_a, pos_a, w_neg)


class DriftQLAgentQ(flax.struct.PyTreeNode):
    """
    Drift Q-learning agent with optional Q-weighted drift (gate + repulsion weighting).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch["next_observations"], seed=sample_rng)
        next_qs = self.network.select("target_critic")(batch["next_observations"], actions=next_actions)

        if self.config["q_agg"] == "min":
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
        q = self.network.select("critic")(batch["observations"], actions=batch["actions"], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def drifting_bc_loss(self, batch, grad_params, rng):
        """
        Train actor_bc_drift with drifting objective:
          For each state s:
            - sample Nneg generated actions a^- ~ actor(s,z)
            - positive is paired dataset action (Npos=1)
            - compute drift V in action space
            - loss = MSE(a^-, stopgrad(clip(a^- + eta * V)))
        plus:
          - optional Q-gating of V magnitude (Iteration 3)
          - optional Q-weighted repulsion targets (Iteration 4)
        """
        batch_size, action_dim = batch["actions"].shape
        rng, noise_rng, sub_rng = jax.random.split(rng, 3)

        drift_bs = int(self.config["drift_batch_size"])
        if drift_bs < batch_size:
            idx = jax.random.choice(sub_rng, batch_size, (drift_bs,), replace=False)
            obs = batch["observations"][idx]
            pos_actions_pool = batch["actions"][idx]
        else:
            obs = batch["observations"]
            pos_actions_pool = batch["actions"]
            drift_bs = batch_size

        pos_actions = pos_actions_pool[:, None, :]  # [B, 1, A]
        pos_actions = jnp.clip(pos_actions, -1.0, 1.0)

        Nneg = int(self.config["drift_nneg"])

        noises = jax.random.normal(noise_rng, (drift_bs * Nneg, action_dim))
        obs_rep = jnp.repeat(obs, repeats=Nneg, axis=0)  # [B*Nneg, ...]
        bc_raw = self.network.select("actor_bc_drift")(obs_rep, noises, params=grad_params)
        bc_actions = bc_raw.reshape(drift_bs, Nneg, action_dim)  # [B, Nneg, A]
        bc_actions = jnp.clip(bc_actions, -1.0, 1.0)

        # =========================================================
        # Q-weighting (gate + repulsion weighting) — toggle via config
        # =========================================================
        q_pos = None
        q_neg = None
        delta_hat = None
        gate = None
        w_neg = None

        q_weight_enable = bool(_cfg_get(self.config, "q_weight_enable", False))
        q_weight_gate = bool(_cfg_get(self.config, "q_weight_gate", True))
        q_weight_repulsion = bool(_cfg_get(self.config, "q_weight_repulsion", True))

        if q_weight_enable:

            def _agg(qs):
                mode = str(_cfg_get(self.config, "q_weight_qagg", "min"))
                return qs.min(axis=0) if mode == "min" else qs.mean(axis=0)

            # Conservative Q(s, a_pos)
            q_pos_ens = self.network.select("target_critic")(obs, actions=pos_actions[:, 0, :])  # [E,B]
            q_pos = _agg(q_pos_ens)  # [B]
            q_pos = jax.lax.stop_gradient(q_pos)

            # Conservative Q(s, a_neg_k)
            a_neg_flat = bc_actions.reshape(drift_bs * Nneg, action_dim)  # [B*Nneg, A]
            q_neg_ens = self.network.select("target_critic")(obs_rep, actions=a_neg_flat)  # [E, B*Nneg]
            q_neg_flat = _agg(q_neg_ens)  # [B*Nneg]
            q_neg = q_neg_flat.reshape(drift_bs, Nneg)  # [B,Nneg]
            q_neg = jax.lax.stop_gradient(q_neg)

            std_eps = float(_cfg_get(self.config, "q_weight_std_eps", 1e-6))
            clipv = float(_cfg_get(self.config, "q_weight_clip", 5.0))

            # Normalize per state for scale robustness
            delta_std = jnp.std(q_neg, axis=1, keepdims=True)  # [B,1]
            delta_std = jax.lax.stop_gradient(delta_std)

            # delta_hat > 0 => dataset action better than this sample
            delta = q_pos[:, None] - q_neg
            delta_hat = delta / (delta_std + std_eps)
            delta_hat = jnp.clip(delta_hat, -clipv, clipv)
            delta_hat = jax.lax.stop_gradient(delta_hat)

            # Iter 3: gate magnitude so high-Q samples move less
            if q_weight_gate:
                beta_g = float(_cfg_get(self.config, "q_weight_gate_beta", 1.0))
                gate = jax.nn.sigmoid(delta_hat / beta_g)  # [B,Nneg]
                gate = jax.lax.stop_gradient(gate)

            # Iter 4: Q-weight repulsion targets (repel mainly from low-Q targets)
            if q_weight_repulsion:
                # Baseline: max(dataset, best-generated) => protects potentially-good unobserved actions
                b = jnp.maximum(q_pos[:, None], jnp.max(q_neg, axis=1, keepdims=True))  # [B,1]
                d = (b - q_neg) / (delta_std + std_eps)  # [B,Nneg], >=0 for worse-than-baseline
                d = jnp.clip(d, 0.0, clipv)
                w_neg = d / (d + 1.0)  # [0,1], w(0)=0
                w_neg = jax.lax.stop_gradient(w_neg)

        temps_cfg = self.config["drift_temps"]
        temps = (float(temps_cfg),) if isinstance(temps_cfg, (int, float)) else tuple(temps_cfg)

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
            gen_a=bc_actions,
            pos_a=pos_actions,
            temps=temps,
            drift_normalize=bool(self.config["drift_normalize"]),
            w_neg=w_neg,
            eps=float(self.config["drift_eps"]),
        )

        # Apply gate AFTER drift compute (affects actual step used)
        if gate is not None:
            V = V * gate[:, :, None]
            drift_pos_total = drift_pos_total * gate[:, :, None]
            drift_neg_total = drift_neg_total * gate[:, :, None]

        eta = float(self.config["drift_eta"])
        target = jax.lax.stop_gradient(jnp.clip(bc_actions + eta * V, -1.0, 1.0))
        bc_drift_loss = jnp.mean((bc_actions - target) ** 2)

        # Diagnostics block (your existing stuff)
        x = bc_actions
        y = pos_actions

        x_drift_preclip = x + eta * V
        x_drift = jnp.clip(x_drift_preclip, -1.0, 1.0)

        drift_clip_frac = jnp.mean((jnp.abs(x_drift_preclip) > 1.0).astype(jnp.float32))
        bc_oob_frac = jnp.mean((jnp.abs(bc_raw) > 1.0).astype(jnp.float32))

        per_sample_pre = jnp.mean((x - y) ** 2, axis=-1)  # [B,Nneg]
        per_sample_post = jnp.mean((x_drift - y) ** 2, axis=-1)  # [B,Nneg]

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
        cos_align = jnp.mean(jnp.sum(V * dir_to_gt, axis=-1) / (v_norm * gt_norm + 1e-12))
        cos_per = jnp.sum(V * dir_to_gt, axis=-1) / (v_norm * gt_norm + 1e-12)
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
            "drift/gt_mse_pre": gt_mse_pre,
            "drift/cos_align_all": cos_align_all,
            "drift/cos_align_p10": cos_align_p10,
            "drift/gt_mse_p10_pre": gt_mse_p10_pre,
            "drift/gt_mse_p10_post": gt_mse_p10_post,
            "drift/gt_mse_p10_improve": gt_mse_p10_improve,
            "drift/gt_mse_min_pre": gt_mse_min_pre,
            "drift/gt_mse_min_post": gt_mse_min_post,
            "drift/gt_mse_min_improve": gt_mse_min_improve,
            "drift/gt_mse_improve": gt_mse_improve,
            "drift/cos_align": cos_align,
            "drift/drift_clip_frac": drift_clip_frac,
            "drift/bc_oob_frac": bc_oob_frac,
            "drift/lam_mean": lam_mean,
            "drift/lam_min": lam_min,
            "drift/lam_max": lam_max,
            "drift/v_raw_sq_mean": v_raw_sq_mean,
            "drift/v_raw_norm_mean": v_raw_norm_mean,
        }

        # Q-weighting diagnostics
        if q_weight_enable and (q_pos is not None):
            info["drift/q_pos_mean"] = jnp.mean(q_pos)
            info["drift/q_neg_mean"] = jnp.mean(q_neg)
            info["drift/delta_hat_mean"] = jnp.mean(delta_hat)
        if gate is not None:
            info["drift/gate_mean"] = jnp.mean(gate)
        if w_neg is not None:
            info["drift/w_neg_mean"] = jnp.mean(w_neg)

        # Per-temp logs (after optional drift_normalize)
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
        """
        actor_loss = alpha * bc_drift_loss + q_loss
        where BOTH losses update actor_bc_drift parameters.
        """
        batch_size, action_dim = batch["actions"].shape
        rng, noise_rng, bc_rng = jax.random.split(rng, 3)

        bc_drift_loss, bc_info = self.drifting_bc_loss(batch, grad_params, bc_rng)

        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        actor_raw = self.network.select("actor_bc_drift")(batch["observations"], noises, params=grad_params)
        actor_raw = jnp.clip(actor_raw, -1.0, 1.0)

        qs = self.network.select("critic")(batch["observations"], actions=actor_raw)

        # IMPORTANT: respect q_agg here (you had mean hardcoded before)
        if self.config["q_agg"] == "min":
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
            network_info["actor_bc_drift_encoder"] = (encoders.get("actor_bc_drift"), (ex_observations,))

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


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="driftql_q",
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
            alpha=10.0,
            drift_eps=1e-12,
            drift_nneg=32,
            drift_npos=1,
            drift_temps=(0.2),
            drift_normalize=True,
            drift_eta=1.0,
            drift_batch_size=256,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
            # -------------------------
            # Q-weighting toggles (Iter 3 & 4)
            # -------------------------
            q_weight_enable=True,      # turn on to compute Q-based weights
            q_weight_gate=True,         # Iter 3: gate drift magnitude by Q-gap
            q_weight_repulsion=True,    # Iter 4: weight repulsion targets by "badness"
            q_weight_qagg="min",        # "min" (safer) or "mean" for weighting only
            q_weight_clip=5.0,
            q_weight_std_eps=1e-6,
            q_weight_gate_beta=1.0,
        )
    )
    return config
