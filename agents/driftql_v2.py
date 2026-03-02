import copy
import os
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


class DriftProjector(nn.Module):
    """
    Projects raw actions into an unbounded feature space
    to prevent near-uniform distances in the bounded action hypercube.
    """
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims[:-1]:
            x = nn.relu(nn.Dense(dim)(x))
        x = nn.Dense(self.hidden_dims[-1])(x)
        return x


def _l2_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _pairwise_l2(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Compute pairwise L2 distance between a [N, F] and b [M, F] -> [N, M].
    Optimized: uses ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
    (avoids materializing [N,M,F]).
    """
    # a: [N,F], b: [M,F]
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
    gen_a: jnp.ndarray,
    pos_a: jnp.ndarray,
    gen_f: jnp.ndarray,
    pos_f: jnp.ndarray,
    temp: float,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Compute drift V for ONE conditioning state (one group),
    using Alg.2 normalization, returning drift vectors in ACTION space.

    Drift uses:
      V = drift_pos - drift_neg
    where drift_pos is weighted sum of pos actions, drift_neg is weighted sum of neg actions.
    """
    Nneg, act_dim = gen_a.shape
    Npos = pos_a.shape[0]

    # Distances are computed in the unbounded feature space
    dist_pos = _pairwise_l2(gen_f, pos_f, eps=eps)
    dist_neg = _pairwise_l2(gen_f, gen_f, eps=eps)

    idx = jnp.arange(Nneg)
    dist_neg = dist_neg.at[idx, idx].set(1e6)

    logit_pos = -dist_pos / temp
    logit_neg = -dist_neg / temp

    logits = jnp.concatenate([logit_pos, logit_neg], axis=-1)
    A = _alg2_attention(logits)

    A_pos = A[:, :Npos]
    A_neg = A[:, Npos:]

    W_pos = A_pos * jnp.sum(A_neg, axis=1, keepdims=True)
    W_neg = A_neg * jnp.sum(A_pos, axis=1, keepdims=True)

    # The actual drift vector V is still constructed in raw action space
    # so we can apply it via: target = bc_actions + eta * V
    drift_pos = W_pos @ pos_a
    drift_neg = W_neg @ gen_a
    V = drift_pos - drift_neg

    return V, drift_pos, drift_neg


def compute_drift_field_conditional(
    gen_a: jnp.ndarray,
    pos_a: jnp.ndarray,
    gen_f: jnp.ndarray,
    pos_f: jnp.ndarray,
    temps: Sequence[float],
    drift_normalize: bool,
    eps: float = 1e-12,
) -> jnp.ndarray:

    def per_item(gen_a_i, pos_a_i, gen_f_i, pos_f_i):
        V_sum = jnp.zeros_like(gen_a_i)
        pos_sum = jnp.zeros_like(gen_a_i)
        neg_sum = jnp.zeros_like(gen_a_i)

        v_norm_list, pos_norm_list, neg_norm_list = [], [], []
        v_sq_list, pos_sq_list, neg_sq_list = [], [], []
        lam_list, v_raw_norm_list, v_raw_sq_list = [], [], []

        for T in temps:
            V_T, dp_T, dn_T = _compute_drift_one_temp(
                gen_a=gen_a_i,
                pos_a=pos_a_i,
                gen_f=gen_f_i,
                pos_f=pos_f_i,
                temp=float(T),
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
            V_sum, pos_sum, neg_sum,
            jnp.stack(v_norm_list, axis=0), jnp.stack(pos_norm_list, axis=0), jnp.stack(neg_norm_list, axis=0),
            jnp.stack(v_sq_list, axis=0), jnp.stack(pos_sq_list, axis=0), jnp.stack(neg_sq_list, axis=0),
            jnp.stack(lam_list, axis=0), jnp.stack(v_raw_norm_list, axis=0), jnp.stack(v_raw_sq_list, axis=0),
        )

    # vmap now includes the new feature arrays
    return jax.vmap(per_item, in_axes=(0, 0, 0, 0))(gen_a, pos_a, gen_f, pos_f)


class DriftQLAgentV2(flax.struct.PyTreeNode):
    """
    - critic_network: Q(s,a)
    - teacher_network (actor_bc_drift): drifting teacher
    - student_network (actor_onestep): student policy
    """
    rng: Any
    critic_network: Any
    teacher_network: Any
    student_network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)

        # Evaluation/bootstrapping uses the student policy
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_qs = self.critic_network.select('target_critic')(batch['next_observations'], actions=next_actions)

        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.critic_network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
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

        drift_bs = int(self.config['drift_batch_size'])
        if drift_bs < batch_size:
            idx = jax.random.choice(sub_rng, batch_size, (drift_bs,), replace=False)
            obs = batch['observations'][idx]
            pos_actions_pool = batch['actions'][idx]
        else:
            obs = batch['observations']
            pos_actions_pool = batch['actions']
            drift_bs = batch_size

        pos_actions = pos_actions_pool[:, None, :]  # [B, 1, A]
        pos_actions = jnp.clip(pos_actions, -1.0, 1.0)

        Nneg = int(self.config['drift_nneg'])
        noises = jax.random.normal(noise_rng, (drift_bs * Nneg, action_dim))
        obs_rep = jnp.repeat(obs, repeats=Nneg, axis=0)  # [B*Nneg, ...]

        bc_raw = self.teacher_network.select('actor_bc_drift')(obs_rep, noises, params=grad_params)
        bc_actions = bc_raw.reshape(drift_bs, Nneg, action_dim)  # [B, Nneg, A]
        bc_actions = jnp.clip(bc_actions, -1.0, 1.0)

        gen_f = self.teacher_network.select('projector')(bc_actions, params=grad_params)
        pos_f = self.teacher_network.select('projector')(pos_actions, params=grad_params)

        temps_cfg = self.config['drift_temps']
        temps = (float(temps_cfg),) if isinstance(temps_cfg, (int, float)) else tuple(temps_cfg)

        (
            V, drift_pos_total, drift_neg_total, v_norms, pos_norms, neg_norms,
            v_sqs, pos_sqs, neg_sqs, lams, v_raw_norms, v_raw_sqs,
        ) = compute_drift_field_conditional(
            gen_a=bc_actions,
            pos_a=pos_actions,
            gen_f=gen_f,
            pos_f=pos_f,
            temps=temps,
            drift_normalize=bool(self.config['drift_normalize']),
            eps=float(self.config['drift_eps']),
        )

        eta = float(self.config['drift_eta'])

        # Stop using jnp.clip on the target and drift application
        target = jax.lax.stop_gradient(bc_actions + eta * V)
        bc_drift_loss = jnp.mean((bc_actions - target) ** 2)

        x = bc_actions
        y = pos_actions
        x_drift = bc_actions + eta * V

        drift_clip_frac = jnp.mean((jnp.abs(x_drift) > 1.0).astype(jnp.float32))
        bc_oob_frac = jnp.mean((jnp.abs(bc_raw.reshape(drift_bs, Nneg, action_dim)) > 1.0).astype(jnp.float32))

        per_sample_pre = jnp.mean((x - y) ** 2, axis=-1)
        per_sample_post = jnp.mean((x_drift - y) ** 2, axis=-1)

        gt_mse_pre = jnp.mean(per_sample_pre)
        gt_mse_post = jnp.mean(per_sample_post)
        gt_mse_improve = gt_mse_pre - gt_mse_post

        gt_mse_min_pre = jnp.mean(jnp.min(per_sample_pre, axis=1))
        gt_mse_min_post = jnp.mean(jnp.min(per_sample_post, axis=1))
        gt_mse_min_improve = gt_mse_min_pre - gt_mse_min_post

        k = jnp.maximum(1, (Nneg * 10) // 100)
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

        lam_mean, lam_min, lam_max = jnp.mean(lams), jnp.min(lams), jnp.max(lams)
        v_raw_sq_mean, v_raw_norm_mean = jnp.mean(v_raw_sqs), jnp.mean(v_raw_norms)

        drift_norm_total = jnp.mean(_l2_norm(V, axis=-1))
        drift_sq_total = jnp.mean(jnp.sum(V * V, axis=-1))
        drift_pos_norm_total = jnp.mean(_l2_norm(drift_pos_total, axis=-1))
        drift_neg_norm_total = jnp.mean(_l2_norm(drift_neg_total, axis=-1))
        drift_pos_sq_total = jnp.mean(jnp.sum(drift_pos_total * drift_pos_total, axis=-1))
        drift_neg_sq_total = jnp.mean(jnp.sum(drift_neg_total * drift_neg_total, axis=-1))

        info = {
            'bc_drift_loss': bc_drift_loss,
            'drift_norm': drift_norm_total,
            'drift_sq_total': drift_sq_total,
            'drift_pos_norm_total': drift_pos_norm_total,
            'drift_neg_norm_total': drift_neg_norm_total,
            'drift_pos_sq_total': drift_pos_sq_total,
            'drift_neg_sq_total': drift_neg_sq_total,
            'npos': jnp.array(pos_actions.shape[1]),
            'nneg': jnp.array(bc_actions.shape[1]),
            'drift/gt_mse_pre': gt_mse_pre,
            'drift/cos_align_all': cos_align_all,
            'drift/cos_align_p10': cos_align_p10,
            'drift/gt_mse_p10_pre': gt_mse_p10_pre,
            'drift/gt_mse_p10_post': gt_mse_p10_post,
            'drift/gt_mse_p10_improve': gt_mse_p10_improve,
            'drift/gt_mse_min_pre': gt_mse_min_pre,
            'drift/gt_mse_min_post': gt_mse_min_post,
            'drift/gt_mse_min_improve': gt_mse_min_improve,
            'drift/cos_align': cos_align,
            'drift/drift_clip_frac': drift_clip_frac,
            'drift/bc_oob_frac': bc_oob_frac,
            'drift/lam_mean': lam_mean,
            'drift/lam_min': lam_min,
            'drift/lam_max': lam_max,
            'drift/v_raw_sq_mean': v_raw_sq_mean,
            'drift/v_raw_norm_mean': v_raw_norm_mean,
        }

        for i, T in enumerate(temps):
            tau_key = f'{float(T):.4f}'.rstrip('0').rstrip('.').replace('.', 'p')
            info[f'drift/tau_{tau_key}/norm'] = jnp.mean(v_norms[:, i])
            info[f'drift/tau_{tau_key}/sq'] = jnp.mean(v_sqs[:, i])
            info[f'drift/tau_{tau_key}/pos_norm'] = jnp.mean(pos_norms[:, i])
            info[f'drift/tau_{tau_key}/neg_norm'] = jnp.mean(neg_norms[:, i])
            info[f'drift/tau_{tau_key}/pos_sq'] = jnp.mean(pos_sqs[:, i])
            info[f'drift/tau_{tau_key}/neg_sq'] = jnp.mean(neg_sqs[:, i])
            info[f'drift/tau_{tau_key}/lam'] = jnp.mean(lams[:, i])
            info[f'drift/tau_{tau_key}/raw_norm'] = jnp.mean(v_raw_norms[:, i])
            info[f'drift/tau_{tau_key}/raw_sq'] = jnp.mean(v_raw_sqs[:, i])

        return bc_drift_loss, info

    def student_loss(self, batch, grad_params, rng):
        """
        Train the separated student policy with Q loss + Distillation.
        """
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, bc_rng = jax.random.split(rng, 3)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))

        # Student policy forward
        student_raw = self.student_network.select('actor_onestep')(batch['observations'], noises, params=grad_params)
        student_actions = jnp.clip(student_raw, -1, 1)

        # Teacher policy forward (stop grad)
        teacher_raw = self.teacher_network.select('actor_bc_drift')(batch['observations'], noises)
        teacher_actions = jax.lax.stop_gradient(jnp.clip(teacher_raw, -1, 1))

        # Distillation loss
        distill_loss = jnp.mean((student_actions - teacher_actions) ** 2)

        # Q loss on student actions
        qs = self.critic_network.select('critic')(batch['observations'], actions=student_actions)

        # Respect Q-aggregation logic for the actor
        if self.config["q_agg_actor"] == "min":
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

        q_loss = -q.mean()

        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
            q_loss = lam * q_loss

        actor_loss = self.config['alpha'] * distill_loss + q_loss

        info = {
            'actor_loss': actor_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
        }

        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, c_rng, t_rng, s_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, c_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        teacher_loss, teacher_info = self.drifting_bc_loss(batch, grad_params, t_rng)
        for k, v in teacher_info.items():
            info[f'teacher/{k}'] = v

        student_loss, student_info = self.student_loss(batch, grad_params, s_rng)
        for k, v in student_info.items():
            info[f'student/{k}'] = v

        loss = critic_loss + teacher_loss + student_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        new_params = flax.core.copy(
            network.params,
            add_or_replace={f'modules_target_{module_name}': new_target_params},
        )
        return network.replace(params=new_params)

    @jax.jit
    def update(self, batch):
        """
        Split updates into independent optimizers.
        """
        new_rng, rng = jax.random.split(self.rng)
        rng, c_rng, t_rng, s_rng = jax.random.split(rng, 4)

        # Update Critic
        def critic_loss_fn(grad_params):
            return self.critic_loss(batch, grad_params, c_rng)
        new_critic, critic_info = self.critic_network.apply_loss_fn(loss_fn=critic_loss_fn)

        # Update Teacher (Drift)
        def teacher_loss_fn(grad_params):
            return self.drifting_bc_loss(batch, grad_params, t_rng)
        new_teacher, teacher_info = self.teacher_network.apply_loss_fn(loss_fn=teacher_loss_fn)

        # Update Student
        def student_loss_fn(grad_params):
            return self.student_loss(batch, grad_params, s_rng)
        new_student, student_info = self.student_network.apply_loss_fn(loss_fn=student_loss_fn)

        new_critic = self.target_update(new_critic, 'critic')

        info = {}
        for k, v in critic_info.items(): info[f'critic/{k}'] = v
        for k, v in teacher_info.items(): info[f'teacher/{k}'] = v
        for k, v in student_info.items(): info[f'student/{k}'] = v

        return self.replace(
            critic_network=new_critic,
            teacher_network=new_teacher,
            student_network=new_student,
            rng=new_rng
        ), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        # Safely fallback if no seed is provided
        if seed is None:
            seed = self.rng

        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )

        raw_actions = self.student_network.select('actor_onestep')(observations, noises)
        actions = jnp.clip(raw_actions, -1.0, 1.0)
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
            encoders['actor_onestep'] = encoder_module()

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

        # Instantiate the student policy
        actor_onestep_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep'),
        )

        # Create isolated ModuleDicts
        critic_mod = ModuleDict({'critic': critic_def, 'target_critic': copy.deepcopy(critic_def)})
        teacher_mod = ModuleDict({'actor_bc_drift': actor_bc_drift_def, 'projector': DriftProjector()})
        student_mod = ModuleDict({'actor_onestep': actor_onestep_def})

        critic_tx = optax.adam(learning_rate=config['lr'])
        teacher_tx = optax.adam(learning_rate=config['lr'])
        student_tx = optax.adam(learning_rate=config['lr'])

        init_rng, c_rng, t_rng, s_rng = jax.random.split(init_rng, 4)

        critic_params = critic_mod.init(
            c_rng,
            critic=(ex_observations, ex_actions),
            target_critic=(ex_observations, ex_actions)
        )['params']

        teacher_params = teacher_mod.init(
            t_rng,
            actor_bc_drift=(ex_observations, ex_actions),
            projector=(ex_actions,)
        )['params']

        student_params = student_mod.init(
            s_rng,
            actor_onestep=(ex_observations, ex_actions)
        )['params']

        critic_network = TrainState.create(critic_mod, critic_params, tx=critic_tx)
        teacher_network = TrainState.create(teacher_mod, teacher_params, tx=teacher_tx)
        student_network = TrainState.create(student_mod, student_params, tx=student_tx)

        # Manually initialize the target critic
        c_params = flax.core.copy(critic_network.params, add_or_replace={'modules_target_critic': critic_network.params['modules_critic']},)
        critic_network = critic_network.replace(params=c_params)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(
            rng,
            critic_network=critic_network,
            teacher_network=teacher_network,
            student_network=student_network,
            config=flax.core.FrozenDict(**config)
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='driftql_v2',
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
            q_agg='min',
            q_agg_actor='mean',
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
        )
    )
    return config
