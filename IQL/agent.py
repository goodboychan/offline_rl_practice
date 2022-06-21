import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense, Dropout

tfd = tfp.distributions
tfb = tfp.bijectors

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.l1 = Dense(256, activation='relu', input_shape=(state_dim, ), kernel_initializer=tf.keras.initializers.Orthogonal(scale=np.sqrt(2)))
        self.l2 = Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(scale=np.sqrt(2)))
        
    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        return x


class Critic(tf.keras.Model):



class IQL(object):
    pass