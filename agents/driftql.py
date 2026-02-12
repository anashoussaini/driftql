# import os
# import copy
# from typing import Any

# import flax
# import jax
# import jax.numpy as jnp
# import ml_collections
# import optax

# from utils.encoders import encoder_modules
# from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
# from utils.networks import ActorVectorField, Value


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# def _l2_norm(x, axis=-1, eps=1e-12):
#     return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


# def compute_drift_field(
#     gen_a: jnp.ndarray,  # [G, A]
#     pos_a: jnp.ndarray,  # [P, A]
#     obs_feat: jnp.ndarray | None,  # [G, S] (and we assume positives share same ordering states in batch)
#     action_temp: float,
#     state_temp: float,
#     use_state: bool,
#     eps: float = 1e-12,
# ):
#     """
#     Batch Monte Carlo estimate of mean-shift drifting field V_{p,q}, producing vectors in action space.

#     Kernel is exp(-||Δ||) over scaled features:
#       feat = [obs_feat / state_temp, action / action_temp] if use_state else [action / action_temp]

#     Drift direction uses action differences (y^+ - y^-), i.e. V is returned in action space.
#     """
#     G, A = gen_a.shape
#     P = pos_a.shape[0]

#     targets_a = jnp.concatenate([gen_a, pos_a], axis=0)  # [G+P, A]

#     if use_state:
#         assert obs_feat is not None
#         # In the offline RL batch, each (s,a) is paired; positives and generated share the same s batch.
#         targets_s = jnp.concatenate([obs_feat, obs_feat], axis=0)  # [G+P, S]
#         gen_feat = jnp.concatenate([obs_feat / state_temp, gen_a / action_temp], axis=-1)  # [G, S+A]
#         tgt_feat = jnp.concatenate([targets_s / state_temp, targets_a / action_temp], axis=-1)  # [G+P, S+A]
#     else:
#         gen_feat = gen_a / action_temp
#         tgt_feat = targets_a / action_temp

#     # cdist(gen_feat, tgt_feat)
#     diff = gen_feat[:, None, :] - tgt_feat[None, :, :]  # [G, G+P, F]
#     dist = _l2_norm(diff, axis=-1, eps=eps)  # [G, G+P]

#     # mask self-match among generated negatives (diagonal in the first G columns)
#     idx = jnp.arange(G)
#     dist = dist.at[idx, idx].set(1e6)

#     kernel = jnp.exp(-dist)  # [G, G+P]

#     # batch-normalized kernel (same spirit as your toy implementation)
#     norm = kernel.sum(axis=-1, keepdims=True) * kernel.sum(axis=-2, keepdims=True)  # [G, G+P]
#     norm = jnp.sqrt(jnp.clip(norm, a_min=eps))
#     K = kernel / norm

#     # Positive and negative components
#     pos_coeff = K[:, G:] * K[:, :G].sum(axis=-1, keepdims=True)  # [G, P]
#     neg_coeff = K[:, :G] * K[:, G:].sum(axis=-1, keepdims=True)  # [G, G]

#     pos_V = pos_coeff @ targets_a[G:]  # [G, A]
#     neg_V = neg_coeff @ targets_a[:G]  # [G, A]
#     return pos_V - neg_V


# class DriftQLAgent(flax.struct.PyTreeNode):
#     """
#     Drift Q-learning agent (FQL-style one-step guidance, but with a drifting BC teacher).

#     Networks:
#       - critic / target_critic: ensemble Q(s,a)
#       - actor_bc_drift: one-step drifting teacher trained with drifting loss
#       - actor_onestep_drift: one-step RL policy trained with Q loss + distillation to teacher
#     """

#     rng: Any
#     network: Any
#     config: Any = nonpytree_field()

#     def critic_loss(self, batch, grad_params, rng):
#         rng, sample_rng = jax.random.split(rng)
#         next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
#         next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)

#         if self.config['q_agg'] == 'min':
#             next_q = next_qs.min(axis=0)
#         else:
#             next_q = next_qs.mean(axis=0)

#         target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
#         q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
#         critic_loss = jnp.square(q - target_q).mean()

#         return critic_loss, {
#             'critic_loss': critic_loss,
#             'q_mean': q.mean(),
#             'q_max': q.max(),
#             'q_min': q.min(),
#         }

#     def drifting_bc_loss(self, batch, grad_params, rng):
#         """
#         Train actor_bc_drift with drifting objective:
#           MSE(gen, stopgrad(gen + V_{p,q}(gen)))
#         """
#         batch_size, action_dim = batch['actions'].shape
#         rng, noise_rng = jax.random.split(rng)

#         noises = jax.random.normal(noise_rng, (batch_size, action_dim))
#         bc_raw = self.network.select('actor_bc_drift')(batch['observations'], noises, params=grad_params)
#         bc_actions = jnp.tanh(bc_raw)  # smooth bounding to [-1, 1]

#         pos_actions = batch['actions']  # dataset actions already in [-1, 1] in this codebase

#         obs_feat = None
#         if self.config['drift_use_state']:
#             if self.config['encoder'] is not None:
#                 obs_feat = self.network.select('actor_bc_drift_encoder')(batch['observations'])
#             else:
#                 obs_feat = batch['observations']

#         # Optionally subsample for O(B^2) kernel cost
#         drift_bs = self.config['drift_batch_size']
#         if drift_bs < batch_size:
#             rng, sub_rng = jax.random.split(rng)
#             idx = jax.random.choice(sub_rng, batch_size, (drift_bs,), replace=False)
#             bc_actions_sub = bc_actions[idx]
#             pos_actions_sub = pos_actions[idx]
#             obs_feat_sub = obs_feat[idx] if obs_feat is not None else None
#         else:
#             bc_actions_sub = bc_actions
#             pos_actions_sub = pos_actions
#             obs_feat_sub = obs_feat

#         V = compute_drift_field(
#             gen_a=bc_actions_sub,
#             pos_a=pos_actions_sub,
#             obs_feat=obs_feat_sub,
#             action_temp=self.config['action_temp'],
#             state_temp=self.config['state_temp'],
#             use_state=self.config['drift_use_state'],
#             eps=self.config['drift_eps'],
#         )

#         target = jax.lax.stop_gradient(jnp.clip(bc_actions_sub + V, -1.0, 1.0))
#         bc_drift_loss = jnp.mean((bc_actions_sub - target) ** 2)

#         return bc_drift_loss, {
#             'bc_drift_loss': bc_drift_loss,
#             'drift_norm': jnp.mean(_l2_norm(V, axis=-1)),
#         }

#     def actor_loss(self, batch, grad_params, rng):
#         """
#         Total actor loss (FQL-shaped):
#           bc_drift_loss + alpha * distill_loss + q_loss
#         """
#         batch_size, action_dim = batch['actions'].shape
#         rng, noise_rng, bc_rng = jax.random.split(rng, 3)

#         # BC drifting teacher loss
#         bc_drift_loss, bc_info = self.drifting_bc_loss(batch, grad_params, bc_rng)

#         # Distillation + Q loss (one-step guidance)
#         noises = jax.random.normal(noise_rng, (batch_size, action_dim))

#         teacher_raw = self.network.select('actor_bc_drift')(batch['observations'], noises)
#         teacher_actions = jnp.tanh(teacher_raw)

#         actor_raw = self.network.select('actor_onestep_drift')(batch['observations'], noises, params=grad_params)
#         actor_actions = jnp.tanh(actor_raw)

#         distill_loss = jnp.mean((actor_actions - teacher_actions) ** 2)

#         qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
#         q = jnp.mean(qs, axis=0)
#         q_loss = -q.mean()

#         if self.config['normalize_q_loss']:
#             lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
#             q_loss = lam * q_loss

#         actor_loss = bc_drift_loss + self.config['alpha'] * distill_loss + q_loss

#         # Useful metric: MSE to dataset actions for sampled actions
#         actions_for_mse = self.sample_actions(batch['observations'], seed=rng)
#         mse = jnp.mean((actions_for_mse - batch['actions']) ** 2)

#         info = {
#             'actor_loss': actor_loss,
#             'bc_drift_loss': bc_info['bc_drift_loss'],
#             'drift_norm': bc_info['drift_norm'],
#             'distill_loss': distill_loss,
#             'q_loss': q_loss,
#             'q': q.mean(),
#             'mse': mse,
#         }
#         return actor_loss, info

#     @jax.jit
#     def total_loss(self, batch, grad_params, rng=None):
#         info = {}
#         rng = rng if rng is not None else self.rng
#         rng, actor_rng, critic_rng = jax.random.split(rng, 3)

#         critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
#         for k, v in critic_info.items():
#             info[f'critic/{k}'] = v

#         actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
#         for k, v in actor_info.items():
#             info[f'actor/{k}'] = v

#         loss = critic_loss + actor_loss
#         return loss, info

#     def target_update(self, network, module_name):
#         new_target_params = jax.tree_util.tree_map(
#             lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
#             self.network.params[f'modules_{module_name}'],
#             self.network.params[f'modules_target_{module_name}'],
#         )
#         network.params[f'modules_target_{module_name}'] = new_target_params

#     @jax.jit
#     def update(self, batch):
#         new_rng, rng = jax.random.split(self.rng)

#         def loss_fn(grad_params):
#             return self.total_loss(batch, grad_params, rng=rng)

#         new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
#         self.target_update(new_network, 'critic')
#         return self.replace(network=new_network, rng=new_rng), info

#     @jax.jit
#     def sample_actions(self, observations, seed=None, temperature=1.0):
#         # One-step policy sampling: z ~ N(0,I), a = tanh(actor(s,z))
#         action_seed, _ = jax.random.split(seed)
#         noises = jax.random.normal(
#             action_seed,
#             (
#                 *observations.shape[: -len(self.config['ob_dims'])],
#                 self.config['action_dim'],
#             ),
#         )
#         raw = self.network.select('actor_onestep_drift')(observations, noises)
#         actions = jnp.tanh(raw)
#         return actions

#     @classmethod
#     def create(cls, seed, ex_observations, ex_actions, config):
#         rng = jax.random.PRNGKey(seed)
#         rng, init_rng = jax.random.split(rng, 2)

#         ob_dims = ex_observations.shape[1:]
#         action_dim = ex_actions.shape[-1]

#         # Define encoders.
#         encoders = dict()
#         if config['encoder'] is not None:
#             encoder_module = encoder_modules[config['encoder']]
#             encoders['critic'] = encoder_module()
#             encoders['actor_bc_drift'] = encoder_module()
#             encoders['actor_onestep_drift'] = encoder_module()

#         # Define networks.
#         critic_def = Value(
#             hidden_dims=config['value_hidden_dims'],
#             layer_norm=config['layer_norm'],
#             num_ensembles=2,
#             encoder=encoders.get('critic'),
#         )

#         actor_bc_drift_def = ActorVectorField(
#             hidden_dims=config['actor_hidden_dims'],
#             action_dim=action_dim,
#             layer_norm=config['actor_layer_norm'],
#             encoder=encoders.get('actor_bc_drift'),
#         )

#         actor_onestep_drift_def = ActorVectorField(
#             hidden_dims=config['actor_hidden_dims'],
#             action_dim=action_dim,
#             layer_norm=config['actor_layer_norm'],
#             encoder=encoders.get('actor_onestep_drift'),
#         )

#         network_info = dict(
#             critic=(critic_def, (ex_observations, ex_actions)),
#             target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
#             actor_bc_drift=(actor_bc_drift_def, (ex_observations, ex_actions)),
#             actor_onestep_drift=(actor_onestep_drift_def, (ex_observations, ex_actions)),
#         )

#         # Make encoder callable for drift kernel state-features
#         if encoders.get('actor_bc_drift') is not None:
#             network_info['actor_bc_drift_encoder'] = (encoders.get('actor_bc_drift'), (ex_observations,))

#         networks = {k: v[0] for k, v in network_info.items()}
#         network_args = {k: v[1] for k, v in network_info.items()}

#         network_def = ModuleDict(networks)
#         network_tx = optax.adam(learning_rate=config['lr'])
#         network_params = network_def.init(init_rng, **network_args)['params']
#         network = TrainState.create(network_def, network_params, tx=network_tx)

#         params = network.params
#         params['modules_target_critic'] = params['modules_critic']

#         config['ob_dims'] = ob_dims
#         config['action_dim'] = action_dim

#         return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# def get_config():
#     config = ml_collections.ConfigDict(
#         dict(
#             agent_name='driftql',  # IMPORTANT: must match key in agents/__init__.py
#             ob_dims=ml_collections.config_dict.placeholder(list),
#             action_dim=ml_collections.config_dict.placeholder(int),
#             lr=3e-4,
#             batch_size=64,
#             actor_hidden_dims=(512, 512, 512, 512),
#             value_hidden_dims=(512, 512, 512, 512),
#             layer_norm=True,
#             actor_layer_norm=False,
#             discount=0.99,
#             tau=0.005,
#             q_agg='mean',
#             # Same role as FQL's BC coefficient alpha: balance behavior regularization vs Q
#             alpha=10.0,
#             # Drift kernel hyperparams
#             drift_use_state=True,
#             action_temp=0.05,
#             state_temp=0.5,
#             drift_eps=1e-12,
#             # If you want faster kernel compute, reduce this (<= batch_size)
#             drift_batch_size=64,
#             normalize_q_loss=False,
#             encoder=ml_collections.config_dict.placeholder(str),  # None, 'impala_small', ...
#         )
#     )
#     return config


import os
import copy
from typing import Any, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def _l2_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _pairwise_l2(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Compute pairwise L2 distance between a [N, F] and b [M, F] -> [N, M].
    """
    diff = a[:, None, :] - b[None, :, :]
    return _l2_norm(diff, axis=-1, eps=eps)


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
    gen_a: jnp.ndarray,      # [Nneg, A]
    pos_a: jnp.ndarray,      # [Npos, A]
    gen_s: jnp.ndarray | None,  # [Nneg, S] (optional)
    pos_s: jnp.ndarray | None,  # [Npos, S] (optional)
    action_temp: float,
    state_temp: float,
    use_state: bool,
    temp: float,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Compute drift V for ONE conditioning state (one group),
    using Alg.2 normalization, returning drift vectors in ACTION space.

    Distances are computed in:
      feat = [s/state_temp, a/action_temp] if use_state else [a/action_temp]

    Drift uses:
      V = drift_pos - drift_neg
    where drift_pos is weighted sum of pos actions, drift_neg is weighted sum of neg actions.
    """
    Nneg, act_dim = gen_a.shape
    Npos = pos_a.shape[0]

    if use_state:
        assert gen_s is not None and pos_s is not None
        gen_feat = jnp.concatenate([gen_s / state_temp, gen_a / action_temp], axis=-1)  # [Nneg, S+A]
        pos_feat = jnp.concatenate([pos_s / state_temp, pos_a / action_temp], axis=-1)  # [Npos, S+A]
        neg_feat = gen_feat  # negatives are the generated set itself
    else:
        gen_feat = gen_a / action_temp
        pos_feat = pos_a / action_temp
        neg_feat = gen_feat

    # Distances
    dist_pos = _pairwise_l2(gen_feat, pos_feat, eps=eps)  # [Nneg, Npos]
    dist_neg = _pairwise_l2(gen_feat, neg_feat, eps=eps)  # [Nneg, Nneg]

    # Mask self in negatives (diagonal)
    idx = jnp.arange(Nneg)
    dist_neg = dist_neg.at[idx, idx].set(1e6)

    # Logits as in paper: -dist / T
    logit_pos = -dist_pos / temp
    logit_neg = -dist_neg / temp

    # Concat for joint normalization across both pos/neg targets
    logits = jnp.concatenate([logit_pos, logit_neg], axis=-1)  # [Nneg, Npos+Nneg]
    A = _alg2_attention(logits)  # [Nneg, Npos+Nneg]

    A_pos = A[:, :Npos]          # [Nneg, Npos]
    A_neg = A[:, Npos:]          # [Nneg, Nneg]

    # Weights from Alg.2
    W_pos = A_pos * jnp.sum(A_neg, axis=1, keepdims=True)  # [Nneg, Npos]
    W_neg = A_neg * jnp.sum(A_pos, axis=1, keepdims=True)  # [Nneg, Nneg]

    drift_pos = W_pos @ pos_a  # [Nneg, A]
    drift_neg = W_neg @ gen_a  # [Nneg, A]

    V = drift_pos - drift_neg
    return V


def compute_drift_field_conditional(
    gen_a: jnp.ndarray,         # [B, Nneg, A]
    pos_a: jnp.ndarray,         # [B, Npos, A]
    obs_feat: jnp.ndarray | None,      # [B, S] (query states)
    pos_obs_feat: jnp.ndarray | None,  # [B, Npos, S] (positive states)
    action_temp: float,
    state_temp: float,
    use_state: bool,
    temps: Sequence[float],
    drift_normalize: bool,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Batched conditional drift:
      - For each batch element i, compute drift in action space for its Nneg generated actions
        relative to Npos positives.

    Supports:
      - Multi-temperature (sum of normalized drifts per temp)
      - Optional drift normalization (paper A.6-style, adapted)
    """

    def per_item(gen_a_i, pos_a_i, s_i, pos_s_i):
        # Expand query state to [Nneg, S] if needed
        if use_state:
            assert s_i is not None and pos_s_i is not None
            gen_s_i = jnp.repeat(s_i[None, :], gen_a_i.shape[0], axis=0)  # [Nneg, S]
        else:
            gen_s_i = None

        V_sum = jnp.zeros_like(gen_a_i)

        for T in temps:
            V_T = _compute_drift_one_temp(
                gen_a=gen_a_i,
                pos_a=pos_a_i,
                gen_s=gen_s_i,
                pos_s=pos_s_i if use_state else None,
                action_temp=action_temp,
                state_temp=state_temp,
                use_state=use_state,
                temp=float(T),
                eps=eps,
            )
            if drift_normalize:
                act_dim = gen_a_i.shape[-1]
                lam = jnp.sqrt(jnp.mean(jnp.sum(V_T * V_T, axis=-1)) / act_dim + eps)
                V_T = V_T / jax.lax.stop_gradient(lam)
            V_sum = V_sum + V_T

        return V_sum

    # If use_state=False we pass None-like placeholders through vmap cleanly
    if use_state:
        assert obs_feat is not None and pos_obs_feat is not None
        V = jax.vmap(per_item, in_axes=(0, 0, 0, 0))(gen_a, pos_a, obs_feat, pos_obs_feat)
    else:
        V = jax.vmap(per_item, in_axes=(0, 0, None, None))(gen_a, pos_a, None, None)

    return V  # [B, Nneg, A]


def _inbatch_knn_indices(x: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    In-batch kNN indices by L2 distance (smallest distance = nearest).
    Returns idx [B, k], includes self (distance 0).
    """
    B = x.shape[0]
    k = int(min(k, B))
    # Pairwise distances [B,B]
    dist = _pairwise_l2(x, x, eps=1e-12)
    sim = -dist
    _, idx = jax.lax.top_k(sim, k)  # top-k similarity => nearest
    return idx


class DriftQLAgent(flax.struct.PyTreeNode):
    """
    Drift Q-learning agent (FQL-style one-step guidance, but with a drifting BC teacher).

    Networks:
      - critic / target_critic: ensemble Q(s,a)
      - actor_bc_drift: one-step drifting teacher trained with drifting loss
      - actor_onestep_drift: one-step RL policy trained with Q loss + distillation to teacher
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
        """
        Train actor_bc_drift with conditional drifting objective:
          For each state s:
            - sample Nneg generated actions a^- ~ q(a|s)
            - get Npos positive actions a^+ ~ pdata(a|s) (approximated in-batch if Npos>1)
            - compute drift V on actions
            - loss = MSE(a^-, stopgrad(a^- + eta * V))
        """
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, sub_rng = jax.random.split(rng, 3)

        # Optional subsample of batch items for drift computation (keeps cost manageable)
        drift_bs = int(self.config['drift_batch_size'])
        if drift_bs < batch_size:
            idx = jax.random.choice(sub_rng, batch_size, (drift_bs,), replace=False)
            obs = batch['observations'][idx]
            pos_actions_pool = batch['actions'][idx]
        else:
            obs = batch['observations']
            pos_actions_pool = batch['actions']
            drift_bs = batch_size

        # Encode observation features if configured
        obs_feat = None
        if self.config['drift_use_state']:
            if self.config['encoder'] is not None:
                # Use the same encoder module as actor_bc_drift (exposed via ModuleDict entry)
                obs_feat = self.network.select('actor_bc_drift_encoder')(obs)
            else:
                obs_feat = obs

            obs_feat = jax.lax.stop_gradient(obs_feat)  # used only for kernels / neighbor selection

        # Build positives per state
        Npos = int(self.config['drift_npos'])
        if (Npos > 1) and (obs_feat is not None):
            nn_idx = _inbatch_knn_indices(obs_feat, Npos)  # [B, Npos]
            pos_actions = jnp.take(pos_actions_pool, nn_idx, axis=0)  # [B, Npos, A]
            pos_obs_feat = jnp.take(obs_feat, nn_idx, axis=0)         # [B, Npos, S]
        else:
            # Fallback: just use the paired dataset action as the only positive
            pos_actions = pos_actions_pool[:, None, :]  # [B, 1, A]
            pos_obs_feat = obs_feat[:, None, :] if obs_feat is not None else None

        # Build Nneg generated actions per state
        Nneg = int(self.config['drift_nneg'])
        # IMPORTANT: Nneg must be >1 for repulsion to work.
        # We won't hard-error here to avoid changing behavior, but performance will suffer if Nneg==1.

        noises = jax.random.normal(noise_rng, (drift_bs * Nneg, action_dim))
        obs_rep = jnp.repeat(obs, repeats=Nneg, axis=0)  # [B*Nneg, ...]
        bc_raw = self.network.select('actor_bc_drift')(obs_rep, noises, params=grad_params)
        bc_actions = jnp.tanh(bc_raw).reshape(drift_bs, Nneg, action_dim)  # [B, Nneg, A]

        V = compute_drift_field_conditional(
            gen_a=bc_actions,
            pos_a=pos_actions,
            obs_feat=obs_feat,
            pos_obs_feat=pos_obs_feat,
            action_temp=float(self.config['action_temp']),
            state_temp=float(self.config['state_temp']),
            use_state=bool(self.config['drift_use_state']),
            temps=tuple(self.config['drift_temps']),
            drift_normalize=bool(self.config['drift_normalize']),
            eps=float(self.config['drift_eps']),
        )

        eta = float(self.config['drift_eta'])
        target = jax.lax.stop_gradient(jnp.clip(bc_actions + eta * V, -1.0, 1.0))
        bc_drift_loss = jnp.mean((bc_actions - target) ** 2)

        return bc_drift_loss, {
            'bc_drift_loss': bc_drift_loss,
            'drift_norm': jnp.mean(_l2_norm(V, axis=-1)),
            'npos': jnp.array(pos_actions.shape[1]),
            'nneg': jnp.array(bc_actions.shape[1]),
        }

    def actor_loss(self, batch, grad_params, rng):
        """
        Total actor loss (FQL-shaped):
          bc_drift_loss + alpha * distill_loss + q_loss
        """
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, bc_rng = jax.random.split(rng, 3)

        # Drifting teacher loss
        bc_drift_loss, bc_info = self.drifting_bc_loss(batch, grad_params, bc_rng)

        # Distillation + Q loss (one-step guidance)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))

        teacher_raw = self.network.select('actor_bc_drift')(batch['observations'], noises)
        teacher_actions = jnp.tanh(teacher_raw)

        actor_raw = self.network.select('actor_onestep_drift')(batch['observations'], noises, params=grad_params)
        actor_actions = jnp.tanh(actor_raw)

        distill_loss = jnp.mean((actor_actions - teacher_actions) ** 2)

        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()

        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
            q_loss = lam * q_loss

        actor_loss = bc_drift_loss + self.config['alpha'] * distill_loss + q_loss

        # Useful metric: MSE to dataset actions for sampled actions
        actions_for_mse = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions_for_mse - batch['actions']) ** 2)

        info = {
            'actor_loss': actor_loss,
            'bc_drift_loss': bc_info['bc_drift_loss'],
            'drift_norm': bc_info['drift_norm'],
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            'npos': bc_info['npos'],
            'nneg': bc_info['nneg'],
        }
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
        # One-step policy sampling: z ~ N(0,I), a = tanh(actor(s,z))
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        raw = self.network.select('actor_onestep_drift')(observations, noises)
        actions = jnp.tanh(raw)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_drift'] = encoder_module()
            encoders['actor_onestep_drift'] = encoder_module()

        # Define networks.
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

        actor_onestep_drift_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_drift'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_drift=(actor_bc_drift_def, (ex_observations, ex_actions)),
            actor_onestep_drift=(actor_onestep_drift_def, (ex_observations, ex_actions)),
        )

        # Make encoder callable for drift kernel state-features
        if encoders.get('actor_bc_drift') is not None:
            network_info['actor_bc_drift_encoder'] = (encoders.get('actor_bc_drift'), (ex_observations,))

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


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='driftql',  # IMPORTANT: must match key in agents/__init__.py
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),

            lr=3e-4,
            batch_size=64,

            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,

            discount=0.99,
            tau=0.005,
            q_agg='mean',

            # Same role as FQL's BC coefficient alpha: balance behavior regularization vs Q
            alpha=10.0,

            # ---- Drift configuration ----
            drift_use_state=True,     # conditional drift (recommended True)
            action_temp=0.05,         # scaling for action component in kernel
            state_temp=0.5,           # scaling for state component in kernel
            drift_eps=1e-12,

            # New (needed) knobs:
            drift_nneg=8,             # Nneg generated actions per state (must be >1 ideally)
            drift_npos=4,             # Npos positives per state (in-batch kNN if drift_use_state=True)
            drift_temps=(0.02, 0.05, 0.2),  # multi-temperature (paper-style)
            drift_normalize=True,     # normalize drift magnitude (paper A.6 style)
            drift_eta=1.0,            # drift step size eta

            # If you want faster drift compute, reduce this (<= batch_size)
            drift_batch_size=64,

            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),  # None, 'impala_small', ...
        )
    )
    return config
