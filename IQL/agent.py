import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import Orthogonal, Zeros

tfd = tfp.distributions
tfb = tfp.bijectors

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, state_dependent_std, log_std_scale, tanh_squash_distribution, temperature):
        super().__init__()
        self.temperature = temperature
        self.tanh_squash_distribution = tanh_squash_distribution

        self.l1 = Dense(256, activation='relu', input_shape=(state_dim, ), kernel_initializer=Orthogonal(scale=np.sqrt(2)))
        self.l2 = Dense(256, activation='relu', kernel_initializer=Orthogonal(scale=np.sqrt(2)))
    
        if not self.tanh_squash_distribution:
            self.means = Dense(action_dim, kernel_initializer=Orthogonal(scale=np.sqrt(2)), activation='tanh')
        else:
            self.means = Dense(action_dim, kernel_initializer=Orthogonal(scale=np.sqrt(2)))            

        if state_dependent_std:
            self.log_stds = Dense(action_dim, kernel_initializer=Orthogonal(scale=log_std_scale))
        else:
            self.log_stds = Dense(action_dim, kernel_initializer=Zeros())
        
    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        log_stds = self.log_stds(x)
        means = self.means(x)
        output = tfd.MultivariateNormalDiag(loc=means, scale_diag=tf.exp(log_stds) * self.temperature)
        if self.tanh_squash_distribution:
            output = tfd.TransformedDistribution(distribution=output, bijector=tfb.Tanh())
        return output

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.l1 = Dense(256, activation='relu', input_shape=(state_dim + action_dim), kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l2 = Dense(256, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l3 = Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.l1(x)
        x = self.l2(x)
        q = self.l3(x)
        return q



class IQL(object):
    def __init__(self, state_dim, action_dim, state_dependent_std, log_std_scale, tanh_squash_distribution):
        self.actor = Actor(state_dim, action_dim, state_dependent_std, log_std_scale, tanh_squash_distribution)
