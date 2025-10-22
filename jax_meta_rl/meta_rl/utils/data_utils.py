import jax
import jax.numpy as jnp
import numpy as np
from flax.training.common_utils import get_metrics, onehot

def collect_trajectories(env, params, policy_fn, num_trajectories, max_steps, rng):
    """
    Collects trajectories from the environment using a given policy.
    """
    trajectories = []
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
        for _ in range(max_steps):
            rng, key = jax.random.split(rng)
            output = policy_fn({'params': params}, obs[None, ...])
            mu, log_std = output[0], output[1]
            std = jnp.exp(log_std)

            # Sample action from Gaussian distribution
            action = mu + jax.random.normal(key, shape=mu.shape) * std
            action = action.squeeze(0) # Remove batch dimension

            # Convert to numpy for env interaction and scale action
            action_np = np.array(action)
            # Pendulum-v1 action space is [-2, 2]
            scaled_action_np = action_np * 2.0

            next_obs, reward, terminated, truncated, _ = env.step(scaled_action_np)
            done = terminated or truncated

            trajectory['observations'].append(obs)
            trajectory['actions'].append(action_np) # Store the unscaled action
            trajectory['rewards'].append(reward)
            trajectory['next_observations'].append(next_obs)
            trajectory['dones'].append(done)

            obs = next_obs
            if done:
                break
        trajectories.append(trajectory)
    return trajectories


def process_trajectories(trajectories, policy_fn, params, gamma=0.99, gae_lambda=0.95):
    """
    Process trajectories to compute advantages using GAE.
    """
    all_advantages = []
    all_observations = []
    all_actions = []
    all_returns = []

    for trajectory in trajectories:
        observations = np.array(trajectory['observations'])
        actions = np.array(trajectory['actions'])
        rewards = np.array(trajectory['rewards'])
        next_observations = np.array(trajectory['next_observations'])
        dones = np.array(trajectory['dones'])

        output = policy_fn({'params': params}, observations)
        values = output[2]
        next_output = policy_fn({'params': params}, next_observations)
        next_values = next_output[2]

        values = np.array(values).squeeze()
        next_values = np.array(next_values).squeeze()

        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_adv = 0
        last_ret = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_adv
            returns[t] = rewards[t] + gamma * (1 - dones[t]) * last_ret
            last_adv = advantages[t]
            last_ret = returns[t]

        all_observations.extend(observations)
        all_actions.extend(actions)
        all_advantages.extend(advantages)
        all_returns.extend(returns)

    return (jnp.array(all_observations),
            jnp.array(all_actions),
            jnp.array(all_advantages),
            jnp.array(all_returns))