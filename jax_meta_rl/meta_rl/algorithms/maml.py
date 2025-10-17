import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Sequence

class ActorCritic(nn.Module):
    """
    An Actor-Critic network that shares parameters between the actor and critic.
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the network.

        Args:
            x: The input observation.

        Returns:
            A tuple containing the logits for the action distribution and the state value.
        """
        h = x
        for hidden_dim in self.hidden_dims:
            h = nn.relu(nn.Dense(features=hidden_dim)(h))

        # Actor head
        mu = nn.Dense(features=self.action_dim)(h)
        mu = nn.tanh(mu) # Squash to [-1, 1]

        # Add a learned standard deviation
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))

        # Critic head
        value = nn.Dense(features=1)(h)

        return mu, log_std, jnp.squeeze(value, axis=-1)

from functools import partial

class MAML:
    def __init__(self,
                 action_dim: int,
                 obs_dim: int,
                 hidden_dims: Sequence[int] = (256, 256),
                 inner_lr: float = 0.1,
                 meta_lr: float = 1e-3):
        """
        Model-Agnostic Meta-Learning (MAML) for Reinforcement Learning.

        Args:
            action_dim: The dimension of the action space.
            obs_dim: The dimension of the observation space.
            hidden_dims: The dimensions of the hidden layers in the ActorCritic network.
            inner_lr: The learning rate for the inner loop (task adaptation).
            meta_lr: The learning rate for the outer loop (meta-update).
        """
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.inner_lr = inner_lr

        # Initialize the Actor-Critic network
        self.network = ActorCritic(action_dim=action_dim, hidden_dims=hidden_dims)

        # Initialize the meta-optimizer
        self.meta_optimizer = optax.adam(meta_lr)

    def init_params(self, rng):
        """Initialize the meta-parameters of the network."""
        obs_sample = jnp.zeros((1, self.obs_dim))
        params = self.network.init(rng, obs_sample)['params']
        opt_state = self.meta_optimizer.init(params)
        return params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _compute_loss(self, params, observations, actions, advantages):
        """Computes the policy gradient loss for continuous actions."""
        mu, log_std, values = self.network.apply({'params': params}, observations)
        std = jnp.exp(log_std)

        # Policy loss (from Gaussian distribution)
        normal_dist = jax.scipy.stats.norm.logpdf(actions, loc=mu, scale=std)
        log_probs = normal_dist.sum(axis=-1)
        policy_loss = -jnp.mean(log_probs * advantages)

        # Value loss
        value_loss = jnp.mean((values - advantages)**2) # Simplified: target is advantages

        return policy_loss + 0.5 * value_loss

    @partial(jax.jit, static_argnums=(0,))
    def inner_update(self, meta_params, task_batch):
        """
        Performs a single inner-loop update for a given task.

        Args:
            meta_params: The current meta-parameters.
            task_batch: A batch of (observations, actions, advantages) for a single task.

        Returns:
            The adapted parameters for the task.
        """
        observations, actions, advantages = task_batch

        # Compute gradients with respect to the loss on the task-specific data
        grad_fn = jax.grad(self._compute_loss)
        grads = grad_fn(meta_params, observations, actions, advantages)

        # Apply one gradient step to get the adapted parameters
        adapted_params = jax.tree_util.tree_map(
            lambda p, g: p - self.inner_lr * g, meta_params, grads
        )
        return adapted_params

    @partial(jax.jit, static_argnums=(0,))
    def outer_update(self, meta_params, opt_state, meta_batch):
        """
        Performs a single outer-loop update (meta-update).

        Args:
            meta_params: The current meta-parameters.
            opt_state: The state of the meta-optimizer.
            meta_batch: A batch of tasks, where each task consists of
                        a support set (for inner update) and a query set (for meta-loss).

        Returns:
            The updated meta-parameters, optimizer state, and the total loss.
        """
        total_loss = 0.0
        meta_grads = jax.tree_util.tree_map(jnp.zeros_like, meta_params)

        # vmap over tasks in the meta-batch
        def update_for_one_task(task_data):
            support_batch, query_batch = task_data

            # Inner update
            adapted_params = self.inner_update(meta_params, support_batch)

            # Compute meta-loss on the query set using adapted parameters
            loss_fn = lambda p: self._compute_loss(p, *query_batch)
            meta_loss, grads = jax.value_and_grad(loss_fn)(adapted_params)

            return meta_loss, grads

        # Use jax.vmap to process all tasks in parallel
        total_loss, meta_grads = jax.vmap(update_for_one_task)(meta_batch)

        # Aggregate gradients and loss
        total_loss = jnp.mean(total_loss)
        meta_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), meta_grads)

        # Update meta-parameters
        updates, new_opt_state = self.meta_optimizer.update(meta_grads, opt_state)
        new_meta_params = optax.apply_updates(meta_params, updates)

        return new_meta_params, new_opt_state, total_loss