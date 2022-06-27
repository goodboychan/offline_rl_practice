import os
# Suppress D4RL import error
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import gym
import d4rl
import numpy as np
from tqdm import tqdm
from absl import flags, app
import tensorflow as tf

import agent, memory

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'halfcheetah-expert-v2', 'Environment name')
flags.DEFINE_integer('seed', 0, 'Seed for random number generators')
flags.DEFINE_float('max_timesteps', 1e6, "Max time steps")
flags.DEFINE_integer("eval_episodes", 10, "Evaluation Episode")
flags.DEFINE_integer('batch_size', 256, "batch size for both actor and critic")
flags.DEFINE_integer('log_freq', int(1e3), 'logging frequency')
flags.DEFINE_float('eval_freq', 100, "evaluation frequency")
flags.DEFINE_float("temperature", 3.0, "temperature")
flags.DEFINE_float("expectile", 0.7, "expectile")
flags.DEFINE_float("tau", 0.005, "tau")
flags.DEFINE_float("discount", 0.99, "Discount factor")

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(state, seed)
            print(action.shape)
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

    # Set seed
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Get Information
    state_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    # Arguments
    kwargs = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'discount': FLAGS.discount,
        'tau': FLAGS.tau,
        'temperature': FLAGS.temperature,
        'expectile': FLAGS.expectile,
        'seed': FLAGS.seed
    }

    policy = agent.IQL(**kwargs)

    replay_buffer = memory.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    mean, std = replay_buffer.normalize_states()

    evaluations = []
    for t in tqdm(range(int(FLAGS.max_timesteps))):
        policy.train(replay_buffer, FLAGS.batch_size)

        if (t + 1) % FLAGS.eval_freq == 0:
            evaluations.append(eval_policy(policy, FLAGS.env, FLAGS.seed, mean, std, eval_episodes=FLAGS.eval_episodes))


if __name__ == '__main__':
    app.run(main)