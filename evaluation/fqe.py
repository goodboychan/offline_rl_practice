# Fitted Q Evaluation (FQE)
# Hyperparameter Selection for Offline Reinforcement Learning
# https://arxiv.org/abs/2007.09055
import copy
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

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

    def Q1(self, state, action):
        x = tf.concat([state, action], axis=-1)

        q1 = self.l1(x)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        return q1


class FQE(object):
    def __init__(self, policy, replay_buffer):
        self.policy = policy
        self.replay_buffer = replay_buffer

    def train_estimator(self, state_dim, action_dim, discount=0.99, target_update_period=100, critic_lr=1e-4, eval_steps=250000, batch_size=256, tau=0.005):
        self.critic = Critic(state_dim, action_dim)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_target = copy.deepcopy(self.critic)
        self.tau = tau

        for t in tqdm(range(eval_steps)):
            batch = self.replay_buffer.sample(batch_size)
            state = batch['state']
            action = batch['action']
            next_state = batch['next_state']
            reward = batch['reward']
            done = batch['done']

            next_action = self.policy.get_action(next_state)
            q_target = self.critic_target(next_state, next_action)
            backup = reward + discount * (1 - done) * q_target
            backup = tf.clip_by_value(backup, -1, 1)

            with tf.GradientTape() as tape:
                q = self.critic(state, action)
                critic_loss = tf.reduce_mean((q - backup) ** 2)

            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            if t % (target_update_period) == 0:
                for var, target_var in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
                    target_var.assign((self.tau * var + (1 - self.tau) * target_var))

        return self.critic