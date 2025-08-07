import numpy as np
import tensorflow as tf

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, context_data_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.context_data = np.zeros((max_size, context_data_dim))

	def add(self, state, action, next_state, reward, done, context_data):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.context_data[self.ptr] = context_data

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			tf.convert_to_tensor(self.state[ind], dtype=tf.float32),
			tf.convert_to_tensor(self.action[ind], dtype=tf.float32),
			tf.convert_to_tensor(self.next_state[ind], dtype=tf.float32),
			tf.convert_to_tensor(self.reward[ind], dtype=tf.float32),
			tf.convert_to_tensor(self.not_done[ind], dtype=tf.float32),
			tf.convert_to_tensor(self.context_data[ind], dtype=tf.float32)
		)

	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		if 'contexts' in dataset:
			self.context_data = dataset['contexts']
		else:
			self.context_data = np.zeros((self.state.shape[0], self.context_data.shape[1]))
		self.size = self.state.shape[0]

	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0, keepdims=True)
		std = self.state.std(0, keepdims=True) + eps
		self.state = (self.state - mean) / std
		self.next_state = (self.next_state - mean) / std
		return mean, std