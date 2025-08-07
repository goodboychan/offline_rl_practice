import numpy as np
import copy
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import Orthogonal, Zeros

tfd = tfp.distributions
tfb = tfp.bijectors

class ContextEncoder(tf.keras.Model):
    def __init__(self, context_dim):
        super().__init__()
        self.l1 = Dense(256, activation='relu')
        self.l2 = Dense(256, activation='relu')
        self.l3 = Dense(context_dim)

    @tf.function
    def call(self, context_data):
        x = self.l1(context_data)
        x = self.l2(x)
        x = self.l3(x)
        return x

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, context_dim, state_dependent_std, log_std_scale, tanh_squash_distribution, temperature, seed):
        super().__init__()
        self.temperature = temperature
        self.tanh_squash_distribution = tanh_squash_distribution
        self.seed = seed

        self.l1 = Dense(256, activation='relu', input_shape=(state_dim + context_dim, ), kernel_initializer=Orthogonal(gain=np.sqrt(2)))
        self.l2 = Dense(256, activation='relu', kernel_initializer=Orthogonal(gain=np.sqrt(2)))

        if not self.tanh_squash_distribution:
            self.means = Dense(action_dim, kernel_initializer=Orthogonal(gain=np.sqrt(2)), activation='tanh')
        else:
            self.means = Dense(action_dim, kernel_initializer=Orthogonal(gain=np.sqrt(2)))

        if state_dependent_std:
            self.log_stds = Dense(action_dim, kernel_initializer=Orthogonal(gain=log_std_scale))
        else:
            self.log_stds = Dense(action_dim, kernel_initializer=Zeros())

    @tf.function
    def call(self, state, context):
        x = tf.concat([state, context], axis=-1)
        x = self.l1(x)
        x = self.l2(x)
        x = tfd.MultivariateNormalDiag(loc=self.means(x), scale_diag=tf.exp(self.log_stds(tf.clip_by_value(x, -10, 2.0))) * self.temperature)
        if self.tanh_squash_distribution:
            x = tfd.TransformedDistribution(distribution=x, bijector=tfb.Tanh())
        return x

    @tf.function
    def sample_action(self, dist, seed):
        return dist.sample(seed=seed)

# Value Critic
class ValueCritic(tf.keras.Model):
    def __init__(self, state_dim, context_dim):
        super().__init__()

        self.l1 = Dense(256, activation='relu', input_shape=(state_dim + context_dim, ), kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l2 = Dense(256, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l3 = Dense(1)

    @tf.function
    def call(self, state, context):
        x = tf.concat([state, context], axis=-1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

# Double Critic
class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, context_dim):
        super().__init__()

        # Q1 Architecture
        self.l1 = Dense(256, activation='relu', input_shape=(state_dim + action_dim + context_dim, ), kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l2 = Dense(256, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l3 = Dense(1)

        # Q2 Architecture
        self.l4 = Dense(256, activation='relu', input_shape=(state_dim + action_dim + context_dim, ), kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l5 = Dense(256, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))
        self.l6 = Dense(1)

    @tf.function
    def call(self, state, action, context):
        x = tf.concat([state, action, context], axis=-1)
        q1 = self.l1(x)
        q1 = self.l2(q1)
        q1 = self.l3(q1)

        q2 = self.l4(x)
        q2 = self.l5(q2)
        q2 = self.l6(q2)
        return q1, q2

    @tf.function
    def Q1(self, state, action, context):
        x = tf.concat([state, action, context], axis=-1)

        x = self.l1(x)
        x = self.l2(x)
        q1 = self.l3(x)
        return q1

# Loss for Expectile Regression
def expectile_loss(diff, expectile=0.8):
    weight = tf.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)

class IQL(object):
    def __init__(self, state_dim, action_dim, context_dim, discount, tau, expectile,
                 temperature, state_dependent_std=False, log_std_scale=1e-3, tanh_squash_distribution=False,
                 actor_lr=3e-4, value_lr=3e-4, critic_lr=3e-4, seed=42):

        self.context_encoder = ContextEncoder(context_dim)
        self.context_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.actor = Actor(state_dim, action_dim, context_dim, state_dependent_std, log_std_scale,
                           tanh_squash_distribution, temperature, seed)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecay(-actor_lr, int(1e6)))

        self.critic = Critic(state_dim, action_dim, context_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.value_critic = ValueCritic(state_dim, context_dim)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_lr)

        self.discount = discount
        self.tau = tau
        self.temperature = temperature

        self.total_it = 0
        self.expectile = expectile

    @tf.function
    def update_v(self, states, actions, context):
        q1, q2 = self.critic_target(states, actions, context)
        q = tf.minimum(q1, q2)

        with tf.GradientTape() as tape:
            v = self.value_critic(states, context)
            value_loss = tf.reduce_mean(expectile_loss(q - v, self.expectile))

        # Optimize Value network model
        grads = tape.gradient(value_loss, self.value_critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_critic.trainable_variables))


    def update_q(self, states, actions, rewards, next_states, dones, context):
        next_v = self.value_critic(next_states, context)
        target_q = (rewards + self.discount * (1 - dones) * next_v)

        with tf.GradientTape() as tape:
            q1, q2 = self.critic(states, actions, context)
            critic_loss = tf.reduce_mean((q1 - target_q) ** 2 + (q2 - target_q) ** 2)

        # Update critic network model
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        grads = [tf.clip_by_norm(g, 1) for g in grads]
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    # Polyak Update
    def update_target(self):
        for var, target_var in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
            target_var.assign((self.tau * var + (1 - self.tau) * target_var))

    # AWR Update
    @tf.function
    def update_actor(self, states, actions, context):
        v = self.value_critic(states, context)
        q1, q2 = self.critic(states, actions, context)
        q = tf.minimum(q1, q2)
        exp_a = tf.exp((q - v) * self.temperature)
        # exp_a = tf.squeeze(tf.clip_by_value(exp_a, exp_a, 100), -1)
        exp_a = tf.minimum(exp_a, 100)

        with tf.GradientTape() as tape:
            dist = self.actor(states, context)
            log_prob = dist.log_prob(actions)
            actor_loss = -tf.reduce_mean(exp_a * log_prob)

        # Update policy network
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def select_action(self, state, context, seed):
        dist = self.actor(state, context)
        return tf.clip_by_value(tf.reshape(self.actor.sample_action(dist, seed), (-1, )), -1, 1)

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        states, actions, next_states, rewards, dones, context_data = replay_buffer.sample(batch_size)

        with tf.GradientTape() as tape:
            context = self.context_encoder(context_data)
            # A placeholder loss for the context encoder, in a real implementation this would be a proper loss function
            context_loss = tf.reduce_mean(context ** 2)

        context_grads = tape.gradient(context_loss, self.context_encoder.trainable_variables)
        self.context_optimizer.apply_gradients(zip(context_grads, self.context_encoder.trainable_variables))

        # Update
        self.update_v(states, actions, context)
        self.update_actor(states, actions, context)
        self.update_q(states, actions, rewards, next_states, dones, context)

        # Soft update
        if self.total_it % 2 == 0:
            self.update_target()