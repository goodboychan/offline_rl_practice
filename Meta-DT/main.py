import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from agent import MetaDecisionTransformer
from memory import MetaDTDataset

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, states, contexts, actions, rewards, dones, rtg, timesteps, masks, prompt):
        state_preds, action_preds, return_preds = self.model.forward(
            states, contexts, actions, rewards, rtg, timesteps, attention_mask=masks, prompt=prompt
        )

        loss = torch.mean((action_preds - actions)**2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Placeholder for environment
    state_dim = 10
    action_dim = 4

    # Placeholder for dataset
    num_trajectories = 50
    max_episode_steps = 100
    trajectories = []
    for _ in range(num_trajectories):
        traj = {
            'observations': np.random.rand(max_episode_steps, state_dim),
            'actions': np.random.rand(max_episode_steps, action_dim),
            'rewards': np.random.rand(max_episode_steps),
            'terminals': np.zeros(max_episode_steps, dtype=bool),
            'contexts': np.random.rand(max_episode_steps, 16) # dummy context
        }
        trajectories.append(traj)

    dataset = MetaDTDataset(
        trajectories,
        horizon=args.dt_horizon,
        max_episode_steps=max_episode_steps,
        return_scale=args.dt_return_scale,
        device=device
    )
    dataloader = DataLoader(dataset, batch_size=args.meta_dt_batch_size, shuffle=True)

    model = MetaDecisionTransformer(
        state_dim=state_dim,
        act_dim=action_dim,
        max_length=args.dt_horizon,
        max_ep_len=max_episode_steps,
        hidden_size=args.dt_embed_dim,
        n_layer=args.dt_n_layer,
        n_head=args.dt_n_head,
        n_inner=4*args.dt_embed_dim,
        activation_function=args.dt_activation_function,
        n_positions=1024,
        resid_pdrop=args.dt_dropout,
        attn_pdrop=args.dt_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.dt_lr,
        weight_decay=args.dt_weight_decay,
    )

    trainer = Trainer(model, optimizer)

    for i, batch in enumerate(dataloader):
        states, contexts, actions, rewards, dones, rtg, timesteps, masks = batch
        rtg = rtg[:, :-1]
        loss = trainer.train_step(states, contexts, actions, rewards, dones, rtg, timesteps, masks, None)
        print(f"Step {i}: loss = {loss}")
        if i >= 10: # run for 10 steps for testing
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt_horizon', type=int, default=20)
    parser.add_argument('--dt_return_scale', type=float, default=1000.0)
    parser.add_argument('--meta_dt_batch_size', type=int, default=128)
    parser.add_argument('--dt_embed_dim', type=int, default=128)
    parser.add_argument('--dt_n_layer', type=int, default=3)
    parser.add_argument('--dt_n_head', type=int, default=1)
    parser.add_argument('--dt_activation_function', type=str, default='relu')
    parser.add_argument('--dt_dropout', type=float, default=0.1)
    parser.add_argument('--dt_lr', type=float, default=1e-4)
    parser.add_argument('--dt_weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
