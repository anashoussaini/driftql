"""
DriftQL RADICAL ABLATION — hardcoded constant behavioral gradient.

No kernels. No repulsion. No pairwise distances. No RMS normalization.
No 32 samples. One generated action per state.

The entire behavioral component is:
    direction = (a+ - â) / ||a+ - â|| * sqrt(d_a)
    target = sg(clip(â + η * direction, -1, 1))
    loss = ||â - target||²

That's it. If this matches DriftQL, the drift framework is not the mechanism.
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
from utils.q_agg import aggregate_q_values

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def _l2_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _cfg_get(cfg, key, default):
    try:
        return cfg[key]
    except KeyError:
        return default


def _resolve_noise_dim(cfg, action_dim):
    noise_dim = int(_cfg_get(cfg, 'noise_dim', action_dim))
    if noise_dim < 0:
        noise_dim = int(action_dim)
    return noise_dim  # 0 = deterministic (no noise input)


class DriftAblationHardcodedAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        next_q = aggregate_q_values(next_qs, self.config['q_agg'])

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng_q, noise_rng_bc = jax.random.split(rng, 3)

        noise_dim = _resolve_noise_dim(self.config, action_dim)
        eta = float(self.config['drift_eta'])
        magnitude = jnp.sqrt(jnp.array(action_dim, dtype=jnp.float32))

        # === BC LOSS: one sample per state, hardcoded magnitude ===
        if noise_dim > 0:
            noises_bc = jax.random.normal(noise_rng_bc, (batch_size, noise_dim))
        else:
            noises_bc = jnp.zeros((batch_size, 1))
        a_hat = self.network.select('actor_bc_drift')(
            batch['observations'], noises_bc, params=grad_params
        )
        a_hat = jnp.clip(a_hat, -1.0, 1.0)

        a_plus = batch['actions']  # [B, A]

        # Direction toward a+
        diff = a_plus - a_hat  # [B, A]
        diff_norm = _l2_norm(diff, axis=-1, eps=1e-12)  # [B]

        # Unit direction * sqrt(d_a)
        V = (diff / diff_norm[:, None]) * magnitude  # [B, A]

        # Stop-gradient target with clipping
        target = jax.lax.stop_gradient(jnp.clip(a_hat + eta * V, -1.0, 1.0))
        bc_loss = jnp.mean((a_hat - target) ** 2)

        # === Q LOSS: separate sample ===
        if noise_dim > 0:
            noises_q = jax.random.normal(noise_rng_q, (batch_size, noise_dim))
        else:
            noises_q = jnp.zeros((batch_size, 1))
        a_hat_q = self.network.select('actor_bc_drift')(
            batch['observations'], noises_q, params=grad_params
        )
        a_hat_q = jnp.clip(a_hat_q, -1.0, 1.0)
        qs = self.network.select('critic')(batch['observations'], actions=a_hat_q)

        q_agg = _cfg_get(self.config, 'q_agg_actor', self.config['q_agg'])
        q = aggregate_q_values(qs, q_agg)
        q_loss = -q.mean()

        if self.config['normalize_q_loss']:
            lam_q = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
            q_loss = lam_q * q_loss

        actor_loss = self.config['alpha'] * bc_loss + q_loss

        # Diagnostics
        actions_for_mse = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions_for_mse - batch['actions']) ** 2)

        info = {
            'actor_loss': actor_loss,
            'bc_drift_loss': bc_loss * self.config['alpha'],
            'bc_drift_loss_raw': bc_loss,
            'drift_norm': jnp.mean(_l2_norm(V, axis=-1)),
            'drift_attract_norm': jnp.mean(_l2_norm(V, axis=-1)),
            'drift_repel_norm': jnp.array(0.0),
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            'drift/cos_align_all': jnp.array(1.0),  # by construction
            'drift/cos_align_p10': jnp.array(1.0),
            'drift/lam_mean': jnp.mean(diff_norm),
            'drift/lam_min': jnp.min(diff_norm),
            'drift/lam_max': jnp.max(diff_norm),
            'drift/gt_mse_pre': jnp.mean(jnp.mean(diff ** 2, axis=-1)),
            'drift/drift_clip_frac': jnp.mean(
                (jnp.abs(a_hat + eta * V) > 1.0).astype(jnp.float32)
            ),
            'drift/bc_oob_frac': jnp.array(0.0),
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
        action_seed, _ = jax.random.split(seed)
        noise_dim = _resolve_noise_dim(self.config, self.config['action_dim'])
        if noise_dim > 0:
            noises = jax.random.normal(
                action_seed,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    noise_dim,
                ),
            )
        else:
            noises = jnp.zeros(
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    1,
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
        noise_dim = _resolve_noise_dim(config, action_dim)
        # For network init, use at least 1 dim even if noise_dim=0 (deterministic)
        net_noise_dim = max(noise_dim, 1)
        ex_noises = jnp.zeros((*ex_actions.shape[:-1], net_noise_dim), dtype=ex_actions.dtype)

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
            actor_bc_drift=(actor_bc_drift_def, (ex_observations, ex_noises)),
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
        config['noise_dim'] = noise_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='drift_ablation_hardcoded',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            noise_dim=0,
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='min',
            q_agg_actor='mean',
            alpha=10.0,
            normalize_q_loss=False,
            drift_eta=1.0,
            drift_eps=1e-12,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config