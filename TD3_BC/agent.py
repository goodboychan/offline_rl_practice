import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor).__init__()

        self.l1 = Dense(256, activation='relu', input_shape=(state_dim,))
        self.l2 = Dense(256, activation='relu')
        self.l3 = Dense(action_dim, activation='tanh')

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        return self.max_action * self.l3(x)

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic).__init__()

        # Q1 Architecture
        self.l1 = Dense(256, activation='relu', input_shape=(state_dim + action_dim,))
        self.l2 = Dense(256, activation='relu')
        self.l3 = Dense(1)

        # Q2 Architecture
        self.l4 = Dense(256, activation='relu', input_shape=(state_dim + action_dim,))
        self.l5 = Dense(256, activation='relu')
        self.l6 = Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        q1 = self.l1(x)
        q1 = self.l2(q1)
        q1 = self.l3(q1)

        q2 = self.l4(x)
        q2 = self.l5(q2)
        q2 = self.l6(q2)
        return q1, q2