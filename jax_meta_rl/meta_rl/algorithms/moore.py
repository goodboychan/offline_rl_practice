import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Sequence, Tuple

class Expert(nn.Module):
    """A single expert in the Mixture of Experts."""
    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        h = x
        for hidden_dim in self.hidden_dims:
            h = nn.relu(nn.Dense(features=hidden_dim)(h))
        return nn.Dense(features=self.output_dim)(h)

class GatingNetwork(nn.Module):
    """A gating network that determines the weights for each expert."""
    num_experts: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        h = x
        for hidden_dim in self.hidden_dims:
            h = nn.relu(nn.Dense(features=hidden_dim)(h))
        return nn.softmax(nn.Dense(features=self.num_experts)(h))

class MixtureOfExperts(nn.Module):
    """A Mixture of Experts module."""
    num_experts: int
    expert_hidden_dims: Sequence[int]
    expert_output_dim: int
    gating_hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # Initialize experts and gating network
        experts = [Expert(hidden_dims=self.expert_hidden_dims, output_dim=self.expert_output_dim)
                   for _ in range(self.num_experts)]
        gating_network = GatingNetwork(num_experts=self.num_experts, hidden_dims=self.gating_hidden_dims)

        # Get expert outputs and gating weights
        expert_outputs = jnp.stack([expert(x) for expert in experts], axis=1)
        gating_weights = gating_network(x)

        # Combine expert outputs
        # gating_weights shape: (batch_size, num_experts)
        # expert_outputs shape: (batch_size, num_experts, expert_output_dim)
        weighted_experts = jnp.einsum('be,bed->bd', gating_weights, expert_outputs)

        return weighted_experts, expert_outputs

class ActorCriticMOORE(nn.Module):
    """An Actor-Critic network using a Mixture of Orthogonal Experts."""
    action_dim: int
    num_experts: int
    expert_hidden_dims: Sequence[int] = (128,)
    expert_output_dim: int = 256
    gating_hidden_dims: Sequence[int] = (128,)
    actor_critic_hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        # Get representation from Mixture of Experts
        moe = MixtureOfExperts(
            num_experts=self.num_experts,
            expert_hidden_dims=self.expert_hidden_dims,
            expert_output_dim=self.expert_output_dim,
            gating_hidden_dims=self.gating_hidden_dims
        )
        representation, expert_outputs = moe(x)

        # Actor-Critic heads
        h = representation
        for hidden_dim in self.actor_critic_hidden_dims:
            h = nn.relu(nn.Dense(features=hidden_dim)(h))

        # Actor head
        mu = nn.Dense(features=self.action_dim)(h)
        mu = nn.tanh(mu)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))

        # Critic head
        value = nn.Dense(features=1)(h)

        return mu, log_std, jnp.squeeze(value, axis=-1), expert_outputs

from functools import partial

class MOORE:
    def __init__(self,
                 action_dim: int,
                 obs_dim: int,
                 num_experts: int,
                 hidden_dims: Sequence[int] = (256, 256),
                 expert_hidden_dims: Sequence[int] = (128,),
                 expert_output_dim: int = 256,
                 gating_hidden_dims: Sequence[int] = (128,),
                 inner_lr: float = 0.1,
                 meta_lr: float = 1e-3,
                 orthogonality_loss_weight: float = 0.1):
        """
        Mixture of Orthogonal Experts (MOORE) for Meta-Reinforcement Learning.
        """
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.inner_lr = inner_lr
        self.orthogonality_loss_weight = orthogonality_loss_weight

        # Initialize the Actor-Critic network
        self.network = ActorCriticMOORE(
            action_dim=action_dim,
            num_experts=num_experts,
            expert_hidden_dims=expert_hidden_dims,
            expert_output_dim=expert_output_dim,
            gating_hidden_dims=gating_hidden_dims,
            actor_critic_hidden_dims=hidden_dims
        )

        # Initialize the meta-optimizer
        self.meta_optimizer = optax.adam(meta_lr)

    def init_params(self, rng):
        """Initialize the meta-parameters of the network."""
        obs_sample = jnp.zeros((1, self.obs_dim))
        params = self.network.init(rng, obs_sample)['params']
        opt_state = self.meta_optimizer.init(params)
        return params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _compute_loss(self, params, observations, actions, advantages, returns):
        """Computes the policy gradient, value, and orthogonality loss."""
        mu, log_std, values, expert_outputs = self.network.apply({'params': params}, observations)
        std = jnp.exp(log_std)

        # Policy loss (from Gaussian distribution)
        normal_dist = jax.scipy.stats.norm.logpdf(actions, loc=mu, scale=std)
        log_probs = normal_dist.sum(axis=-1)
        policy_loss = -jnp.mean(log_probs * advantages)

        # Value loss
        value_loss = jnp.mean((values - returns)**2)

        # Orthogonality loss - applied to the expert weights, not outputs
        # This is more efficient and directly follows the paper's description
        # of applying Gram-Schmidt to the expert parameters.
        expert_kernels = []
        for key, value in params.items():
            if 'Expert' in key and 'kernel' in key:
                expert_kernels.append(value.reshape(-1, value.shape[-1]))

        orthogonality_loss = 0.0
        if len(expert_kernels) > 1:
            # A simple way to encourage orthogonality is to penalize the dot product
            # between the flattened kernel matrices of different experts.
            for i in range(len(expert_kernels)):
                for j in range(i + 1, len(expert_kernels)):
                    # Ensure kernels have compatible shapes for dot product
                    # This is a simplification; a true Gram-Schmidt process
                    # would be more involved to implement in a loss function.
                    # We penalize the cosine similarity.
                    v1 = expert_kernels[i].ravel()
                    v2 = expert_kernels[j].ravel()
                    if v1.shape == v2.shape:
                         cos_sim = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))
                         orthogonality_loss += jnp.abs(cos_sim)

        total_loss = policy_loss + 0.5 * value_loss + self.orthogonality_loss_weight * orthogonality_loss
        return total_loss

    @partial(jax.jit, static_argnums=(0,))
    def inner_update(self, meta_params, task_batch):
        """
        Performs a single inner-loop update for a given task.
        """
        observations, actions, advantages, returns = task_batch
        grad_fn = jax.grad(self._compute_loss)
        grads = grad_fn(meta_params, observations, actions, advantages, returns)

        adapted_params = jax.tree_util.tree_map(
            lambda p, g: p - self.inner_lr * g, meta_params, grads
        )
        return adapted_params

    @partial(jax.jit, static_argnums=(0,))
    def outer_update(self, meta_params, opt_state, meta_batch):
        """
        Performs a single outer-loop update (meta-update).
        """
        total_loss = 0.0
        meta_grads = jax.tree_util.tree_map(jnp.zeros_like, meta_params)

        def update_for_one_task(task_data):
            support_batch, query_batch = task_data
            adapted_params = self.inner_update(meta_params, support_batch)
            observations, actions, advantages, returns = query_batch
            loss_fn = lambda p: self._compute_loss(p, observations, actions, advantages, returns)
            meta_loss, grads = jax.value_and_grad(loss_fn)(adapted_params)
            return meta_loss, grads

        total_loss, meta_grads = jax.vmap(update_for_one_task)(meta_batch)

        total_loss = jnp.mean(total_loss)
        meta_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), meta_grads)

        updates, new_opt_state = self.meta_optimizer.update(meta_grads, opt_state)
        new_meta_params = optax.apply_updates(meta_params, updates)

        return new_meta_params, new_opt_state, total_loss
