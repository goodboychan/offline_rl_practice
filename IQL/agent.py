import numpy as np
import copy
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import Orthogonal, Zeros

tfd = tfp.distributions
tfb = tfp.bijectors

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, state_dependent_std, log_std_scale, tanh_squash_distribution, temperature, seed):
        super().__init__()
        self.temperature = temperature
        self.tanh_squash_distribution = tanh_squash_distribution
        self.seed = seed

        self.l1 = Dense(256, activation='relu', input_shape=(state_dim, ), kernel_initializer=Orthogonal(gain=np.sqrt(2)))
        self.l2 = Dense(256, activation='relu', kernel_initializer=Orthogonal(gain=np.sqrt(2)))
    
        if not self.tanh_squash_distribution:
            self.means = Dense(action_dim, kernel_initializer=Orthogonal(gain=np.sqrt(2)), activation='tanh')
        else:
            self.means = Dense(action_dim, kernel_initializer=Orthogonal(gain=np.sqrt(2)))            

        if state_dependent_std:
            self.log_stds = Dense(action_dim, kernel_initializer=Orthogonal(gain=log_std_scale))
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

    def sample_action(self, seed):
        return self.dist.sample(seed)

# Value Critic
class ValueCritic(tf.keras.Model):
    def __init__(self, state_dim):
        super().__init__()

        self.l1 = Dense(256, activation='relu', input_shape=(state_dim, ), kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l2 = Dense(256, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l3 = Dense(1)

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        return x

# Double Critic 
class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Q1 Architecture
        self.l1 = Dense(256, activation='relu', input_shape=(state_dim + action_dim, ), kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l2 = Dense(256, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l3 = Dense(1)

        # Q2 Architecture
        self.l4 = Dense(256, activation='relu', input_shape=(state_dim + action_dim, ), kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l5 = Dense(256, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))
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

        x = self.l1(x)
        x = self.l2(x)
        q1 = self.l3(x)
        return q1

# Loss for Expectile Regression
def expectile_loss(diff, expectile=0.8):
    weight = tf.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)

class IQL(object):
    def __init__(self, state_dim, action_dim, discount, tau, expectile, 
                 temperature, state_dependent_std=True, log_std_scale=1.0, tanh_squash_distribution=True, 
                 actor_lr=3e-4, value_lr=3e-4, critic_lr=3e-4, seed=42):
        self.actor = Actor(state_dim, action_dim, state_dependent_std, log_std_scale, 
                           tanh_squash_distribution, temperature, seed)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecay(-actor_lr, int(1e6)))
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.value_critic = ValueCritic(state_dim)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_lr)

        self.discount = discount
        self.tau = tau
        self.temperature = temperature

        self.total_it = 0
        self.expectile = expectile

    def update_v(self, states, actions):
        q1, q2 = self.critic_target(states, actions)
        q = tf.minimum(q1, q2)

        with tf.GradientTape() as tape:
            v = self.value_critic(states)
            value_loss = tf.reduce_mean(expectile_loss(q - v, self.expectile))

        # Optimize Value network model
        grads = tape.gradient(value_loss, self.value_critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_critic.trainable_variables))

    def update_q(self, states, actions, rewards, next_states, dones):
        next_v = self.value_critic(next_states)
        target_q = (rewards + self.discount * (1 - dones) * next_v)

        with tf.GradientTape() as tape:
            q1, q2 = self.critic(states, actions)
            critic_loss = tf.reduce_mean((q1 - target_q) ** 2 + (q2 - target_q) ** 2)

        # Update critic network model
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def update_target(self):
        for var, target_var in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
            target_var.assign((self.tau * var + (1 - self.tau) * target_var))

    # AWR Update
    def update_actor(self, states, actions):
        v = self.value_critic(states)
        q1, q2 = self.critic(states, actions)
        q = tf.minimum(q1, q2)
        exp_a = tf.exp((q - v) * self.temperature)
        exp_a = tf.clip_by_value(exp_a, exp_a, 100)

        with tf.GradientTape() as tape:
            dist = self.actor(states)
            log_prob = dist.log_prob(actions)
            actor_loss = -tf.reduce_mean(exp_a * log_prob)
        
        # Update policy network
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def select_action(self, state, seed):
        state = tf.reshape(state, (1, -1))
        dist = self.actor(state)
        return self.actor.sample_action(seed)

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        # print(action.shape)

        # Update
        self.update_v(state, action)
        self.update_actor(state, action)
        self.update_q(state, action, reward, next_state, not_done)
        self.update_target()

    


        
        