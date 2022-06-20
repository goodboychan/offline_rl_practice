import os
# Suppress D4RL import error
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import gym
import d4rl
import tensorflow as tf
import numpy as np

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'hopper-medium-v0', 'Environment name')
flags.DEFINE_integer('seed', 0, 'Seed for random number generators')

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

    
    


if __name__ == '__main__':
    app.run(main)