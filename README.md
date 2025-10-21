# Offline RL Practice
A set of offline RL algorithm with tensorflow-keras. 
- Use [D4RL](https://github.com/rail-berkeley/d4rl) dataset
- For the purpose of result reproducibility.
- Heavily borrow most test concepts from TD3+BC.

## TD3+BC 
A Minimalist Approach to Offline RL, Fujumoto et al [[paper](https://arxiv.org/abs/2106.06860)], [[repo](https://github.com/sfujim/TD3_BC)]

## Implicit Q-Learning (IQL)
Offline Reinforcement Learning with Implicit Q-Learning, Kostrikov et al [[paper](https://arxiv.org/abs/2110.06169)][[repo(jax)](https://github.com/ikostrikov/implicit_q_learning)][[repo(pytorch)](https://github.com/BY571/Implicit-Q-Learning)]

## JAX Meta-RL (OMOS)
A JAX-based implementation of "Offline Meta-Reinforcement Learning with Online Self-Supervision" (OMOS).

### Setup with `uv`
This project uses `uv` for fast and unified package management.

1.  **Install `uv`**:
    ```bash
    pip install uv
    ```

2.  **Create virtual environment and install dependencies**:
    ```bash
    uv venv
    uv pip install -r requirements.txt
    ```
    This will create a `.venv` directory and install all necessary packages.

3.  **Activate the environment**:
    ```bash
    source .venv/bin/activate
    ```

### Running Experiments
You can find example notebooks in the `notebooks/` directory. For example, to run the OMOS experiment on the MetaWorld MT50 benchmark:

1.  **Generate the dataset**:
    ```bash
    uv run python scripts/generate_dataset.py
    ```

2.  **Run the notebook**:
    ```bash
    uv run jupyter notebook notebooks/run_omos_mt50.ipynb
    ```
