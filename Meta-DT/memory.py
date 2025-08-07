import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class MetaDTDataset(Dataset):
    def __init__(self, trajectories, horizon, max_episode_steps, return_scale, device):
        self.trajectories = trajectories
        self.horizon = horizon
        self.max_episode_steps = max_episode_steps
        self.return_scale = return_scale
        self.device = device

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())

        self.traj_lens = np.array(traj_lens)
        self.returns = np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        num_timesteps = sum(self.traj_lens)

        print(f'Dataset info: {len(trajectories)} trajectories, {num_timesteps} transitions')
        print(f'Returns: min = {self.returns.min()}, max = {self.returns.max()}, avg = {self.returns.mean()}')

        self.parse_trajectory_segment()

    def parse_trajectory_segment(self):
        states, actions, rewards, dones, rtg, timesteps, masks = [], [], [], [], [], [], []

        for traj in self.trajectories:
            for si in range(traj['rewards'].shape[0]):
                # get sequences from dataset
                state_seg = traj['observations'][si : si+self.horizon]
                action_seg = traj['actions'][si : si+self.horizon]
                reward_seg = traj['rewards'][si : si+self.horizon].reshape(-1, 1)

                if 'terminals' in traj:
                    done_seg = traj['terminals'][si : si+self.horizon].reshape(-1)
                else:
                    done_seg = traj['dones'][si : si+self.horizon].reshape(-1)

                timestep_seg = np.arange(si, si+state_seg.shape[0]).reshape(-1)
                timestep_seg[timestep_seg >= self.max_episode_steps] = self.max_episode_steps - 1  # padding cutoff

                rtg_seg = discount_cumsum(traj['rewards'][si:], gamma=1.)[:state_seg.shape[0] + 1].reshape(-1, 1)
                if rtg_seg.shape[0] <= state_seg.shape[0]:
                    rtg_seg = np.concatenate([rtg_seg, np.zeros((1, 1))], axis=0)

                # padding and state + reward normalization
                tlen = state_seg.shape[0]
                state_seg = np.concatenate([np.zeros((self.horizon - tlen, state_seg.shape[1])), state_seg], axis=0)
                state_seg = (state_seg - self.state_mean) / self.state_std

                action_seg = np.concatenate([np.ones((self.horizon - tlen, action_seg.shape[1])) * -10., action_seg], axis=0)
                reward_seg = np.concatenate([np.zeros((self.horizon - tlen, 1)), reward_seg], axis=0)
                done_seg = np.concatenate([np.ones((self.horizon - tlen)) * 2, done_seg], axis=0)
                rtg_seg = np.concatenate([np.zeros((self.horizon - tlen, 1)), rtg_seg], axis=0) / self.return_scale
                timestep_seg = np.concatenate([np.zeros((self.horizon - tlen)), timestep_seg], axis=0)
                mask_seg = np.concatenate([np.zeros((self.horizon - tlen)), np.ones((tlen))], axis=0)

                states.append(state_seg)
                actions.append(action_seg)
                rewards.append(reward_seg)
                dones.append(done_seg)
                rtg.append(rtg_seg)
                timesteps.append(timestep_seg)
                masks.append(mask_seg)

        self.states = torch.from_numpy(np.stack(states, axis=0)).to(dtype=torch.float32, device=self.device)
        self.actions = torch.from_numpy(np.stack(actions, axis=0)).to(dtype=torch.float32, device=self.device)
        self.rewards = torch.from_numpy(np.stack(rewards, axis=0)).to(dtype=torch.float32, device=self.device)
        self.dones = torch.from_numpy(np.stack(dones, axis=0)).to(dtype=torch.long, device=self.device)
        self.rtg = torch.from_numpy(np.stack(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        self.timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        self.masks = torch.from_numpy(np.stack(masks, axis=0)).to(dtype=torch.float32, device=self.device)

    def __getitem__(self, index):
        return (
            self.states[index],
            torch.zeros_like(self.states[index][:, :16]), # dummy context
            self.actions[index],
            self.rewards[index],
            self.dones[index],
            self.rtg[index],
            self.timesteps[index],
            self.masks[index]
        )

    def __len__(self):
        return self.states.size(0)
