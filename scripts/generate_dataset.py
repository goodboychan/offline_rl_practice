import metaworld
import metaworld.policies as policies
import numpy as np
import pickle
import os

def generate_mt50_dataset(num_trajectories_per_task=10, trajectory_length=200, seed=42):
    """
    Generates an offline dataset for the MetaWorld MT50 benchmark by rolling out
    a combination of expert and random policies.
    """
    mt50 = metaworld.MT50(seed=seed)
    dataset = {}

    for name, env_cls in mt50.train_classes.items():
        print(f"Generating data for task: {name}")
        task_dataset = []

        # Use a skilled policy for the specific task if available
        # This provides more meaningful data than a purely random policy.
        policy_name = name.replace('-', '_') + '_policy'
        try:
            policy = getattr(policies, policy_name)
        except AttributeError:
            print(f"  - No expert policy found for {name}. Using random policy.")
            policy = None # Fallback to random actions

        for i in range(num_trajectories_per_task):
            env = env_cls()
            task = [t for t in mt50.train_tasks if t.env_name == name][0]
            env.set_task(task)

            obs, _ = env.reset()
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'terminals': [],
            }

            for _ in range(trajectory_length):
                trajectory['observations'].append(obs)

                if policy:
                    action = policy.get_action(obs)
                    # Add some noise for exploration
                    action += np.random.normal(0, 0.2, size=action.shape)
                    action = np.clip(action, -1.0, 1.0)
                else:
                    action = env.action_space.sample()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['next_observations'].append(next_obs)
                trajectory['terminals'].append(done)

                obs = next_obs
                if done:
                    obs, _ = env.reset()

            # Convert lists to numpy arrays
            for key in trajectory:
                trajectory[key] = np.array(trajectory[key], dtype=np.float32)

            task_dataset.append(trajectory)

        dataset[name] = task_dataset

    # Save the dataset
    os.makedirs('data', exist_ok=True)
    with open('data/mt50_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    print("\nDataset generation complete. Saved to data/mt50_dataset.pkl")

if __name__ == '__main__':
    generate_mt50_dataset()
