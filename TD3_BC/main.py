import os
# Suppress D4RL import error
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import gym
import d4rl
import tensorflow as tf
import numpy as np
import agent
import memory
from tqdm import tqdm
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'walker2d-medium-v0', 'Environment name')
flags.DEFINE_integer('seed', 0, 'Seed for random number generators')
flags.DEFINE_float('max_timesteps', 1e6, "Max time steps")
flags.DEFINE_integer('batch_size', 256, "batch size for both actor and critic")
flags.DEFINE_float('eval_freq', 5e3, "evaluation frequency")

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score

def main(argv):
    env = gym.make(FLAGS.env)

    # Set seeds
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Argments
    kwags = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'discount': 0.99,
        'tau': 0.005,
        'policy_noise': 0.2 * max_action,
        'noise_clip': 0.5 * max_action,
        'policy_freq': 2,
        'alpha': 2.5,
    }

    policy = agent.TD3_BC(**kwags)

    replay_buffer = memory.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    mean, std = replay_buffer.normalize_states()

    evaluations = []
    for t in tqdm(range(int(FLAGS.max_timesteps))):
        policy.train(replay_buffer, FLAGS.batch_size)
        if (t + 1) % FLAGS.eval_freq == 0:
            evaluations.append(eval_policy(policy, FLAGS.env, FLAGS.seed, mean, std))
    


if __name__ == '__main__':
    app.run(main)