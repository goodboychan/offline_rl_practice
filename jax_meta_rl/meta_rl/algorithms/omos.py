import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class SharedEncoder(nn.Module):
    """A shared encoder for observations."""
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations):
        x = observations
        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(features=hidden_dim)(x))
        return x

class Actor(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, encoded_obs):
        mu = nn.Dense(features=self.action_dim)(encoded_obs)
        mu = nn.tanh(mu)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        return mu, log_std

class Critic(nn.Module):
    @nn.compact
    def __call__(self, encoded_obs, actions):
        x = jnp.concatenate([encoded_obs, actions], axis=-1)
        q_value = nn.Dense(features=1)(x)
        return jnp.squeeze(q_value, axis=-1)

class DoubleCritic(nn.Module):
    def setup(self):
        self.critic1 = Critic()
        self.critic2 = Critic()

    def __call__(self, encoded_obs, actions):
        return self.critic1(encoded_obs, actions), self.critic2(encoded_obs, actions)

class ForwardDynamics(nn.Module):
    obs_dim: int
    @nn.compact
    def __call__(self, encoded_obs, actions):
        x = jnp.concatenate([encoded_obs, actions], axis=-1)
        next_obs_delta = nn.Dense(features=self.obs_dim)(x)
        return next_obs_delta

import optax
from functools import partial

class OMOS:
    """Offline Meta-RL with Online Self-Supervision (OMOS)."""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dims: Sequence[int] = (256, 256),
                 inner_lr: float = 0.01,
                 meta_lr: float = 1e-4,
                 cql_alpha: float = 5.0,
                 dynamics_lr: float = 1e-4,
                 tau: float = 0.005):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.cql_alpha = cql_alpha
        self.tau = tau

        self.encoder = SharedEncoder(hidden_dims=hidden_dims)
        self.actor = Actor(action_dim=action_dim)
        self.critic = DoubleCritic()
        self.target_critic = DoubleCritic()
        self.dynamics_model = ForwardDynamics(obs_dim=obs_dim)

        # Optimizers
        self.optimizer = optax.adam(meta_lr)
        self.dynamics_optimizer = optax.adam(dynamics_lr)

    def init_params(self, rng):
        rng, encoder_rng, actor_rng, critic_rng, dynamics_rng = jax.random.split(rng, 5)

        obs_sample = jnp.zeros((1, self.obs_dim))
        action_sample = jnp.zeros((1, self.action_dim))

        encoder_params = self.encoder.init(encoder_rng, obs_sample)['params']
        encoded_sample = self.encoder.apply({'params': encoder_params}, obs_sample)

        actor_params = self.actor.init(actor_rng, encoded_sample)['params']
        critic_params = self.critic.init(critic_rng, encoded_sample, action_sample)['params']
        dynamics_params = self.dynamics_model.init(dynamics_rng, encoded_sample, action_sample)['params']

        params = {
            'encoder': encoder_params,
            'actor': actor_params,
            'critic': critic_params,
            'dynamics': dynamics_params,
            'target_critic': critic_params.copy()
        }

        opt_state = self.optimizer.init(params)
        dynamics_opt_state = self.dynamics_optimizer.init({'encoder': params['encoder'], 'dynamics': params['dynamics']})

        return params, {'main': opt_state, 'dynamics': dynamics_opt_state}

    @partial(jax.jit, static_argnums=(0,))
    def _compute_cql_loss(self, params, batch, rng):
        obs, actions, rewards, next_obs, dones = batch

        encoded_obs = self.encoder.apply({'params': params['encoder']}, obs)
        encoded_next_obs = self.encoder.apply({'params': params['encoder']}, next_obs)

        rng, next_action_key, rand_action_key = jax.random.split(rng, 3)

        next_mu, next_log_std = self.actor.apply({'params': params['actor']}, encoded_next_obs)
        next_std = jnp.exp(next_log_std)
        next_actions = next_mu + jax.random.normal(next_action_key, shape=next_mu.shape) * next_std

        next_q1, next_q2 = self.target_critic.apply({'params': params['target_critic']}, encoded_next_obs, next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        target_q = rewards + (1. - dones) * 0.99 * next_q

        q1, q2 = self.critic.apply({'params': params['critic']}, encoded_obs, actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        random_actions = jax.random.uniform(rand_action_key, actions.shape, minval=-1.0, maxval=1.0)
        q1_rand, q2_rand = self.critic.apply({'params': params['critic']}, encoded_obs, random_actions)
        cql_penalty = (q1_rand - q1).mean() + (q2_rand - q2).mean()

        return critic_loss + self.cql_alpha * cql_penalty

    @partial(jax.jit, static_argnums=(0,))
    def _compute_actor_loss(self, params, batch, rng):
        obs, _, _, _, _ = batch
        encoded_obs = self.encoder.apply({'params': params['encoder']}, obs)
        mu, log_std = self.actor.apply({'params': params['actor']}, encoded_obs)
        q1, _ = self.critic.apply({'params': params['critic']}, encoded_obs, mu)
        return -q1.mean()

    @partial(jax.jit, static_argnums=(0,))
    def inner_update(self, params, task_batch, rng):

        def critic_loss_fn(p):
            return self._compute_cql_loss(p, task_batch, rng)

        critic_grads = jax.grad(critic_loss_fn)(params)

        adapted_params = jax.tree_util.tree_map(
            lambda p, g: p - self.inner_lr * g, params, critic_grads
        )

        def actor_loss_fn(p):
            return self._compute_actor_loss(p, task_batch, rng)

        actor_grads = jax.grad(actor_loss_fn)(adapted_params)

        adapted_params = jax.tree_util.tree_map(
            lambda p, g: p - self.inner_lr * g, adapted_params, actor_grads
        )

        return adapted_params

    @partial(jax.jit, static_argnums=(0,))
    def outer_update(self, params, opt_state, meta_batch, rng):

        def meta_loss_for_task(p, task_data, key):
            support_batch, query_batch = task_data
            adapted_params = self.inner_update(p, support_batch, key)

            critic_loss = self._compute_cql_loss(adapted_params, query_batch, key)
            actor_loss = self._compute_actor_loss(adapted_params, query_batch, key)

            return critic_loss + actor_loss

        grad_fn = jax.grad(meta_loss_for_task)

        # Create a unique key for each task in the meta-batch
        rngs = jax.random.split(rng, jax.tree_util.tree_leaves(meta_batch)[0].shape[0])
        meta_grads_batch = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, meta_batch, rngs)

        meta_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), meta_grads_batch)

        updates, new_opt_state = self.optimizer.update(meta_grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        # For loss reporting
        losses = jax.vmap(meta_loss_for_task, in_axes=(None, 0, 0))(params, meta_batch, rngs)
        total_loss = jnp.mean(losses)

        # Update target critic params
        new_params['target_critic'] = jax.tree_util.tree_map(
            lambda target, online: target * (1 - self.tau) + online * self.tau,
            params['target_critic'], new_params['critic']
        )

        return new_params, new_opt_state, total_loss

    @partial(jax.jit, static_argnums=(0,))
    def self_supervised_update(self, params, opt_states, online_batch):
        obs, actions, next_obs = online_batch

        def dynamics_loss_fn(p):
            encoded_obs = self.encoder.apply({'params': p['encoder']}, obs)
            predicted_delta = self.dynamics_model.apply({'params': p['dynamics']}, encoded_obs, actions)
            true_delta = next_obs - obs
            return ((predicted_delta - true_delta)**2).mean()

        dynamics_params_to_update = {'encoder': params['encoder'], 'dynamics': params['dynamics']}
        dynamics_loss, grads = jax.value_and_grad(dynamics_loss_fn)(dynamics_params_to_update)

        updates, new_dynamics_opt_state = self.dynamics_optimizer.update(grads, opt_states['dynamics'])
        updated_dynamics_params = optax.apply_updates(dynamics_params_to_update, updates)

        new_params = params.copy()
        new_params['encoder'] = updated_dynamics_params['encoder']
        new_params['dynamics'] = updated_dynamics_params['dynamics']

        new_opt_states = opt_states.copy()
        new_opt_states['dynamics'] = new_dynamics_opt_state

        return new_params, new_opt_states, dynamics_loss
