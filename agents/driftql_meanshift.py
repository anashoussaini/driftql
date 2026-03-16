"""
DriftQL agent — clean implementation based on the mean-shift drifting framework.

The drift field is:
    Δ_{p,q}(x) = V_{p,k}(x) - V_{q,k}(x)

where V_{π,k}(x) = Σ_j w_j (y_j - x) is the kernel-weighted mean-shift direction,
with w_j = softmax(-d_j / τ) being the normalized kernel weights.

For Gaussian kernel: this is EXACTLY score matching on smoothed distributions (Theorem 1).
For Laplace kernel: this APPROXIMATES score matching with error O(1/D) (Theorems 4-6).

Reference: "Drifting models train one-step generators by optimizing a mean-shift
discrepancy..." (arXiv:2603.07514, March 2026)
"""

import copy
import os
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


# =============================================================================
# Utilities
# =============================================================================


def _l2_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _pairwise_l2(a, b, eps=1e-12):
    """Pairwise L2: a [N,F], b [M,F] -> [N,M]."""
    a2 = jnp.sum(a * a, axis=-1, keepdims=True)
    b2 = jnp.sum(b * b, axis=-1, keepdims=True)
    dist2 = a2 + b2.T - 2.0 * (a @ b.T)
    return jnp.sqrt(jnp.maximum(dist2, 0.0) + eps)


def _pairwise_l2_sq(a, b, eps=1e-12):
    """Pairwise squared L2: a [N,F], b [M,F] -> [N,M]."""
    a2 = jnp.sum(a * a, axis=-1, keepdims=True)
    b2 = jnp.sum(b * b, axis=-1, keepdims=True)
    dist2 = a2 + b2.T - 2.0 * (a @ b.T)
    return jnp.maximum(dist2, 0.0) + eps


def _cfg_get(cfg, key, default):
    try:
        return cfg[key]
    except KeyError:
        return default


# =============================================================================
# Mean-Shift Drift Field
#
# This computes the paper's Δ_{p,q}(x) = V_{p,k}(x) - V_{q,k}(x)
# where V_{π,k}(x) = Σ_j w_j(x) (y_j - x)
# and w_j(x) = k(x, y_j) / Σ_l k(x, y_l)  [softmax of logits]
#
# Kernel choices:
#   Laplace:  k(x,y) = exp(-||x-y|| / τ)    → logit = -d / τ
#   Gaussian: k(x,y) = exp(-||x-y||² / 2τ²) → logit = -d² / 2τ²
#
# Per the paper (Theorem 1), Gaussian gives EXACT score matching:
#   V_{π,k}(x) = τ² ∇log π_τ(x)
# Laplace gives approximate score matching with O(1/D) error.
# =============================================================================


def _compute_mean_shift_drift(
    gen_a: jnp.ndarray,  # [Ngen, A] — generated actions (= q samples)
    pos_a: jnp.ndarray,  # [Npos, A] — dataset actions (= p samples)
    temp: float,
    kernel: str = 'laplace',  # "laplace" or "gaussian"
    dim_scale: bool = True,  # divide distances by sqrt(d_a)
    eps: float = 1e-12,
):
    """
    Compute the mean-shift drift field for one state.

    Returns:
        V:          [Ngen, A] — total drift = V_p - V_q
        V_attract:  [Ngen, A] — attraction component V_p(x)
        V_repel:    [Ngen, A] — repulsion component V_q(x)
    """
    Ngen, action_dim = gen_a.shape

    # Pairwise distances
    dist_pos = _pairwise_l2(gen_a, pos_a, eps=eps)  # [Ngen, Npos]
    dist_neg = _pairwise_l2(gen_a, gen_a, eps=eps)  # [Ngen, Ngen]

    # Self-mask: each sample cannot repel itself
    idx = jnp.arange(Ngen)
    dist_neg = dist_neg.at[idx, idx].set(1e6)

    # Dimension-aware scaling (paper: τ = τ̄ · D^a with a ≈ 1/2)
    if dim_scale:
        dim_factor = jnp.sqrt(jnp.asarray(action_dim, dtype=gen_a.dtype))
    else:
        dim_factor = 1.0

    # Kernel logits
    if kernel == 'gaussian':
        # Gaussian: k(x,y) = exp(-||x-y||² / 2τ²)
        # This gives EXACT score matching (Theorem 1)
        dist_pos_scaled = dist_pos / dim_factor
        dist_neg_scaled = dist_neg / dim_factor
        logit_pos = -(dist_pos_scaled**2) / (2.0 * temp * temp)
        logit_neg = -(dist_neg_scaled**2) / (2.0 * temp * temp)
    else:
        # Laplace: k(x,y) = exp(-||x-y|| / τ)
        # Approximates score matching with O(1/D) error (Theorems 4-6)
        logit_pos = -(dist_pos / dim_factor) / temp
        logit_neg = -(dist_neg / dim_factor) / temp

    # Separate softmax — V_p and V_q are independently normalized
    # This is the paper's definition: w_j = k(x,y_j) / Σ_l k(x,y_l)
    W_pos = jax.nn.softmax(logit_pos, axis=-1)  # [Ngen, Npos]
    W_neg = jax.nn.softmax(logit_neg, axis=-1)  # [Ngen, Ngen]

    # Displacement vectors (y - x), the paper's mean-shift definition
    disp_to_pos = pos_a[None, :, :] - gen_a[:, None, :]  # [Ngen, Npos, A]
    disp_to_neg = gen_a[None, :, :] - gen_a[:, None, :]  # [Ngen, Ngen, A]

    # Mean-shift: V_{π,k}(x) = Σ_j w_j (y_j - x)
    V_attract = jnp.sum(W_pos[:, :, None] * disp_to_pos, axis=1)  # [Ngen, A]
    V_repel = jnp.sum(W_neg[:, :, None] * disp_to_neg, axis=1)  # [Ngen, A]

    # Drift = attraction - repulsion (paper's Δ_{p,q})
    V = V_attract - V_repel

    return V, V_attract, V_repel


def _compute_drift_batched(
    gen_a: jnp.ndarray,  # [B, Ngen, A]
    pos_a: jnp.ndarray,  # [B, Npos, A]
    temp: float,
    kernel: str = 'laplace',
    dim_scale: bool = True,
    drift_normalize: bool = False,
    eps: float = 1e-12,
):
    """
    Batched drift field computation with diagnostics.

    Returns:
        V:          [B, Ngen, A]
        V_attract:  [B, Ngen, A]
        V_repel:    [B, Ngen, A]
        lam:        [B] — RMS magnitude (for logging)
        raw_norm:   [B] — mean L2 norm before any normalization
    """

    def per_item(gen_i, pos_i):
        V_i, Va_i, Vr_i = _compute_mean_shift_drift(
            gen_a=gen_i,
            pos_a=pos_i,
            temp=temp,
            kernel=kernel,
            dim_scale=dim_scale,
            eps=eps,
        )

        act_dim = gen_i.shape[-1]
        raw_sq = jnp.mean(jnp.sum(V_i * V_i, axis=-1)) / act_dim
        raw_norm = jnp.mean(_l2_norm(V_i, axis=-1))
        lam = jax.lax.stop_gradient(jnp.sqrt(raw_sq + eps))

        if drift_normalize:
            V_i = V_i / lam
            Va_i = Va_i / lam
            Vr_i = Vr_i / lam

        return V_i, Va_i, Vr_i, lam, raw_norm

    return jax.vmap(per_item)(gen_a, pos_a)


# =============================================================================
# Agent
# =============================================================================


class DriftQLMeanShiftAgent(flax.struct.PyTreeNode):
    """
    DriftQL with theoretically-grounded mean-shift drift field.

    The drift field follows the drifting model framework exactly:
      Δ_{p,q}(x) = V_{p,k}(x) - V_{q,k}(x)

    where V_{π,k}(x) is the kernel-weighted mean-shift direction.
    With Gaussian kernel, this is exact score matching on smoothed distributions.
    With Laplace kernel, it approximates score matching with O(1/D) error.

    Training loss:
      L_actor = α · L_drift + L_Q
      L_drift = ||â - sg(clip(â + η·V, -1, 1))||²
      L_Q     = -E[Q(s, f_θ(s,z))]
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)

        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def drifting_bc_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, sub_rng = jax.random.split(rng, 3)

        # Read config
        kernel = str(_cfg_get(self.config, 'kernel', 'laplace'))
        dim_scale = bool(_cfg_get(self.config, 'dim_scale', True))
        drift_normalize = bool(self.config['drift_normalize'])
        temp = float(self.config['drift_temp'])
        eta = float(self.config['drift_eta'])
        Ngen = int(self.config['drift_ngen'])

        # Optional subsample for drift computation
        drift_bs = int(self.config['drift_batch_size'])
        if drift_bs < batch_size:
            idx = jax.random.choice(sub_rng, batch_size, (drift_bs,), replace=False)
            obs = batch['observations'][idx]
            pos_actions = batch['actions'][idx]
        else:
            obs = batch['observations']
            pos_actions = batch['actions']
            drift_bs = batch_size

        # Positive actions: dataset actions for each state [B, 1, A]
        pos_a = jnp.clip(pos_actions[:, None, :], -1.0, 1.0)

        # Generate Ngen samples per state from current policy
        noises = jax.random.normal(noise_rng, (drift_bs * Ngen, action_dim))
        obs_rep = jnp.repeat(obs, repeats=Ngen, axis=0)
        bc_raw = self.network.select('actor_bc_drift')(obs_rep, noises, params=grad_params)
        gen_a = jnp.clip(bc_raw.reshape(drift_bs, Ngen, action_dim), -1.0, 1.0)

        # Compute drift field: Δ_{p,q}(x) = V_p(x) - V_q(x)
        V, V_attract, V_repel, lam, raw_norm = _compute_drift_batched(
            gen_a=gen_a,
            pos_a=pos_a,
            temp=temp,
            kernel=kernel,
            dim_scale=dim_scale,
            drift_normalize=drift_normalize,
            eps=float(self.config['drift_eps']),
        )

        # Drift loss: paper's Equation 3
        # L = ||f_θ(ε) - sg(f_θ(ε) + η·Δ_{p,q}(f_θ(ε)))||²
        target = jax.lax.stop_gradient(jnp.clip(gen_a + eta * V, -1.0, 1.0))
        bc_drift_loss = jnp.mean((gen_a - target) ** 2)

        # ---- Diagnostics ----
        x = gen_a
        y = pos_a
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

        # Cosine alignment: does V point toward the expert?
        dir_to_gt = y - x
        v_norm_val = _l2_norm(V, axis=-1, eps=1e-12)
        gt_norm_val = _l2_norm(dir_to_gt, axis=-1, eps=1e-12)
        cos_per = jnp.sum(V * dir_to_gt, axis=-1) / (v_norm_val * gt_norm_val + 1e-12)
        cos_align_all = jnp.mean(cos_per)

        # Top-10% alignment
        Ngen_local = per_sample_pre.shape[1]
        k = jnp.maximum(1, (Ngen_local * 10) // 100)
        sorted_pre = jnp.sort(per_sample_pre, axis=1)
        p10_thresh = sorted_pre[:, k - 1][:, None]
        mask = (per_sample_pre <= p10_thresh).astype(jnp.float32)
        cos_align_p10 = jnp.sum(cos_per * mask) / (jnp.sum(mask) + 1e-12)

        # Attraction / repulsion balance
        attract_norm = jnp.mean(_l2_norm(V_attract, axis=-1))
        repel_norm = jnp.mean(_l2_norm(V_repel, axis=-1))
        drift_norm = jnp.mean(_l2_norm(V, axis=-1))

        info = {
            'bc_drift_loss': bc_drift_loss,
            'drift_norm': drift_norm,
            'drift_attract_norm': attract_norm,
            'drift_repel_norm': repel_norm,
            'npos': jnp.array(pos_a.shape[1]),
            'ngen': jnp.array(Ngen),
            'drift/kernel': jnp.array(1 if kernel == 'gaussian' else 0),
            'drift/gt_mse_pre': gt_mse_pre,
            'drift/gt_mse_post': gt_mse_post,
            'drift/gt_mse_improve': gt_mse_improve,
            'drift/gt_mse_min_pre': gt_mse_min_pre,
            'drift/gt_mse_min_post': gt_mse_min_post,
            'drift/gt_mse_min_improve': gt_mse_min_pre - gt_mse_min_post,
            'drift/cos_align_all': cos_align_all,
            'drift/cos_align_p10': cos_align_p10,
            'drift/drift_clip_frac': drift_clip_frac,
            'drift/bc_oob_frac': bc_oob_frac,
            'drift/lam_mean': jnp.mean(lam),
            'drift/lam_min': jnp.min(lam),
            'drift/lam_max': jnp.max(lam),
            'drift/raw_norm_mean': jnp.mean(raw_norm),
        }

        return bc_drift_loss, info

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, bc_rng = jax.random.split(rng, 3)

        bc_drift_loss, bc_info = self.drifting_bc_loss(batch, grad_params, bc_rng)

        # Q-loss: maximize Q(s, f_θ(s,z))
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        actor_raw = self.network.select('actor_bc_drift')(batch['observations'], noises, params=grad_params)
        actor_raw = jnp.clip(actor_raw, -1.0, 1.0)
        qs = self.network.select('critic')(batch['observations'], actions=actor_raw)

        q_agg = _cfg_get(self.config, 'q_agg_actor', self.config['q_agg'])
        if q_agg == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

        q_loss = -q.mean()

        if self.config['normalize_q_loss']:
            lam_q = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
            q_loss = lam_q * q_loss

        actor_loss = self.config['alpha'] * bc_drift_loss + q_loss

        # MSE diagnostic
        actions_for_mse = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions_for_mse - batch['actions']) ** 2)

        info = {
            'actor_loss': actor_loss,
            'bc_drift_loss': bc_info['bc_drift_loss'] * self.config['alpha'],
            'bc_drift_loss_raw': bc_info['bc_drift_loss'],
            'drift_norm': bc_info['drift_norm'],
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
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
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        raw_actions = self.network.select('actor_bc_drift')(observations, noises)
        actions = jnp.clip(raw_actions, -1.0, 1.0)
        actions = jnp.where(jnp.isnan(actions), 0.0, actions)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_drift'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )

        actor_bc_drift_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_drift'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_drift=(actor_bc_drift_def, (ex_observations, ex_actions)),
        )

        if encoders.get('actor_bc_drift') is not None:
            network_info['actor_bc_drift_encoder'] = (
                encoders.get('actor_bc_drift'),
                (ex_observations,),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# =============================================================================
# Config
# =============================================================================


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent
            agent_name='driftql_meanshift',
            # Placeholders (filled at runtime)
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            # Optimizer
            lr=3e-4,
            batch_size=256,
            # Architecture
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            # Critic
            discount=0.99,
            tau=0.005,  # Polyak target update rate
            q_agg='min',  # "min" or "mean" for critic target
            q_agg_actor='mean',  # "min" or "mean" for actor Q-loss
            # Actor
            alpha=10.0,  # drift loss weight
            normalize_q_loss=False,
            # ------------------------------------------
            # Drift field (the mean-shift from the paper)
            # ------------------------------------------
            kernel='laplace',  # "laplace" or "gaussian"
            #   gaussian = exact score matching (Theorem 1)
            #   laplace  = approx score matching, O(1/D) error
            drift_temp=0.2,  # kernel temperature τ
            dim_scale=True,  # divide distances by sqrt(d_a) before applying τ
            #   corresponds to paper's τ = τ̄ · D^{1/2}
            drift_ngen=32,  # number of generated samples per state
            drift_eta=1.0,  # drift step size η
            drift_normalize=True,  # normalize drift RMS for stable actor/Q balancing
            drift_batch_size=256,  # subsample size for drift computation
            drift_eps=1e-12,
            # Encoder
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
