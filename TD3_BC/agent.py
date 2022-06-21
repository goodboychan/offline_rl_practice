import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self.l1 = Dense(256, activation='relu', input_shape=(state_dim,))
        self.l2 = Dense(256, activation='relu')
        self.l3 = Dense(action_dim, activation='tanh')

        self.max_action = max_action

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        return self.max_action * self.l3(x)

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

class TD3_BC(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, alpha=2.5):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

    def select_action(self, state):
        state = tf.reshape(state, (1, -1))
        return tf.reshape(self.actor(state), [-1])
    
    # @tf.function
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = tf.random.normal(shape=action.shape)
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)

        next_action = self.actor_target(next_state) + noise
        next_action = tf.clip_by_value(next_action, -self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = tf.minimum(target_Q1, target_Q2)
        target_Q = reward + not_done * self.discount * target_Q

        with tf.GradientTape() as tape:
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = tf.keras.losses.mean_squared_error(target_Q, current_Q1) + tf.keras.losses.mean_squared_error(target_Q, current_Q2)

        # Optimize Critic
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            with tf.GradientTape() as tape:
                pi = self.actor(state)
                Q = self.critic.Q1(state, pi)
                lmbda = self.alpha / tf.math.reduce_mean(tf.abs(Q))

                actor_loss = - lmbda * tf.reduce_mean(Q) + tf.keras.losses.mean_squared_error(pi, action)

            # Optimize the actor
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            # Update the frozen target models
            for var, target_var in zip(self.actor.trainable_variables, self.actor_target.trainable_variables):
                target_var.assign((self.tau * var + (1 - self.tau) * target_var))

            for var, target_var in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
                target_var.assign((self.tau * var + (1 - self.tau) * target_var))





            