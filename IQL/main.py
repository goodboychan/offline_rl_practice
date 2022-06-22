import os
# Suppress D4RL import error
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import gym
import d4rl
import numpy as np
import tqdm
from absl import flags, app

import memory

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'halfcheetah-expert-v2', 'Environment name')
flags.DEFINE_integer('seed', 0, 'Seed for random number generators')
flags.DEFINE_float('max_timesteps', 1e6, "Max time steps")
flags.DEFINE_integer('batch_size', 256, "batch size for both actor and critic")
flags.DEFINE_integer('log_freq', int(1e3), 'logging frequency')
flags.DEFINE_float('eval_freq', 5e3, "evaluation frequency")

def main(argv):
    env = gym.make(FLAGS.env)

    # Set seed
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)

    # Get Information
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = memory.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))


if __name__ == '__main__':
    app.run(main)