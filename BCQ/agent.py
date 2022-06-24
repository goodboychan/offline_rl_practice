import copy
from sre_constants import LITERAL_UNI_IGNORE
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class FC_Q(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q1
        self.l1 = Dense(256, activation='relu', input_shape=(state_dim, ))
        self.l2 = Dense(256, activation='relu')
        self.l3 = Dense(action_dim, activation='relu')

        # Q2
        self.l4 = Dense(256, activation='relu', input_shape=(state_dim, ))
        self.l5 = Dense(256, activation='relu')
        self.l6 = Dense(action_dim, activation='relu')

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        q = self.l3(x)

        x = self.l4(state)
        x = self.l5(x)
        x = self.l6(x)
        imt = tf.nn.log_softmax(x, axis=1)
        return q, imt, x

class discrete_BCQ(object):
    def __init__(self, state_dim, action_dim, BCQ_threshold=0.3, discount=0.99, 
                 polyak_target_update=False, target_update_frequency=8e3, tau=0.005,
                 initial_eps=1, end_eps=0.001, eps_decay_period=25e4, eval_eps=0.001):
        self.Q = FC_Q(state_dim, action_dim)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.discount = discount
        self.maybe_update_target = polyak_target_update 
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for epsilon
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyperparameter
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.action_dim = action_dim

        self.threshold = BCQ_threshold
        self.total_it = 0

    def select_action(self, state, eval=False):
        # Select action according to policy with probability (1-eps)
		# otherwise, select random action
        if np.random.uniform(0,1) > self.eval_eps:
            state = tf.reshape(state, -1)
            q, imt, i = self.Q(state)
            imt = tf.exp(imt)
            imt = tf.cast((imt / tf.reduce_max(imt, 1, keepdims=True)[0] > self.threshold), 'float')
            return tf.argmax(imt * q + (1. - imt) * -1e8, 1).numpy()
        else:
            np.random.randint(self.action_dim)

    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done =replay_buffer.sample()

        q, imt, i = self.Q(next_state)
        imt = tf.exp(imt)
        imt = tf.cast((imt / tf.reduce_max(imt, 1, keepdims=True)[0] > self.threshold), 'float')

        # Use large negative number to mask actions from argmax
        next_action = tf.argmax((imt * q + (1. - imt) * -1e8), 1)

        q, imt, i = self.Q_target(next_state)
        target_Q = reward + done * self.discount * tf.gather(q, next_action, axis=1).reshape(-1, 1)

        with tf.GradientTape() as tape:
            # Get current Q estimate
            current_Q, imt, i = self.Q(state)
            current_Q = tf.gather(current_Q, action, axis=1)

            # Compute Q loss
            q_loss = tf.keras.losses.Huber()(current_Q, target_Q)
            i_loss = tf.keras.losses.SparseCategoricalCrossEntropy(imt, tf.reshape(action, -1))

            Q_loss = q_loss + i_loss + 1e-2 * tf.reduce_mean(tf.pow(i, 2))

        grads = tape.gradient(Q_loss, self.Q.trainable_variables)
        self.Q_optimizer.apply_gradients(zip(grads, self.Q.trainable_variables))

        self.total_it += 1
        
        for var, target_var in zip(self.Q.trainable_variables, self.Q_target.trainable_variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
