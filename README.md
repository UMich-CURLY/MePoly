# **M**ax **E**ntropy **Poly**nomial Policy Optimization

<!-- Official code release for the paper:
**"MePoly: Max Entropy Polynomial Policy Optimization"** -->

![](.media/fig-bandit.png)

This repository provides a unified training and evaluation framework for continuous-action policy optimization with:
- `MePoly` (max-entropy polynomial policy)
- `PPO` (Gaussian policy)
- `FPO` (flow/diffusion-based policy optimization)
- `GMM-PPO` (Gaussian mixture policy)

across two environments:
- **Bandit**: 1-step 2D distribution fitting
- **SmoothWorld**: continuous 2D navigation with obstacles/goals

## Highlights

- Unified codebase for multiple policy families under a PPO-style training pipeline
- Max-entropy polynomial policy with Legendre/monomial basis support
- Diffusion/flow-based policy variant (FPO) with CFM-based surrogate objective
- Reproducible multi-seed benchmark scripts for both environments
- Built-in visualization and Weights & Biases logging

## Repository Layout

```text
.
|-- bandit/
|   |-- main.py
|   |-- models/
|   |   |-- ppo.py
|   |   |-- fpo.py
|   |   |-- mepoly.py
|   |   |-- gmm.py
|   |   |-- diffusion_policy.py
|   |   `-- network.py
|   `-- utils/
|       `-- bandit.py
|-- smoothworld/
|   |-- main.py
|   |-- envs/
|   |   `-- smoothworld.py
|   `-- models/
|       |-- ppo.py
|       |-- fpo.py
|       |-- mepoly.py
|       |-- gmm.py
|       |-- diffusion_policy.py
|       `-- network.py
|-- bandit_bench.sh
`-- smoothworld_bench.sh
```

## Installation

### 1. Create environment

```bash
conda create -n mepoly python=3.10 -y
conda activate mepoly
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib gymnasium wandb torchdiffeq ipdb
```

Optional:

```bash
pip install -r smoothworld/requirements.txt
```

### 3. Configure logging 
strongly recommended, since our evaluation will upload to wandb
```bash
wandb login
# All visualization result will be uploaded to wandb for quick checking!!
```

## Quick Start

### Bandit (1-step distribution fitting)

```bash
cd bandit
SEED=$RANDOM

# MePoly
python bandit/main.py --method mepoly --task lemniscate --ent_coef=0.2 --poly_order=14 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}

# PPO (Gaussian)
python bandit/main.py --method ppo --task lemniscate --ent_coef=0.2  --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}

# FPO (flow/diffusion policy)
python bandit/main.py --method fpo --task lemniscate --n_updates_per_iteration=10 --num_fpo_samples=2000 --max_iterations=100 --seed=${SEED}

# GMM-PPO
python bandit/main.py --method gmm --task lemniscate --ent_coef=0.1  --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
```

Available tasks:
- `lemniscate`
- `twomoons`

In case you are interesting to reproduce our experiments and ablation, check `bandit_bench.sh` for detailed hyperparameters. Or you can directly check our training log at  [Bandit-wandb-log](https://wandb.ai/1160476985/bandit-dist-fit)

### SmoothWorld (continuous navigation)

```bash
cd smoothworld
SEED=$RANDOM

# MePoly
python smoothworld/main.py --method mepoly --poly_order=5 --ent_coef=0.10 --lambda_clip=5 --max_iterations=2000 --poly_grid_size=64 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=four_walls

# PPO
python smoothworld/main.py --method ppo --ent_coef=0.0010 --max_iterations=2000 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=four_walls

# FPO
python smoothworld/main.py --method fpo --num_fpo_samples=50 --max_iterations=2000 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=four_walls

# GMM-PPO
python smoothworld/main.py --method gmm --ent_coef=0.0010 --max_iterations=2000 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=four_walls
```

Available grid modes:
- `two_walls`
- `three_goals`
- `tree_in_the_middle`
- `two_slits`
- `cshape`
- `four_walls`

In case you are interesting to reproduce our experiments and ablation, check `smoothworld_bench.sh` for detailed hyperparameters. Or you can directly check our training log at  [smoothworld-wandb-log](https://wandb.ai/1160476985/smoothworld?nw=nwuser1160476985)


## Reproducing Benchmarks

Top-level benchmark scripts include multi-seed experiment grids:

```bash
bash bandit_bench.sh
bash smoothworld_bench.sh
```

The scripts run paper-style sweeps over methods/tasks with random seeds.

## Training Outputs

Model checkpoints are saved under:
- `bandit/run/<method>/`
- `smoothworld/run/<method>/`

Typical checkpoint files:
- `<method>_actor.pth`
- `<method>_critic.pth`
- `<method>_actor_<iteration>.pth`
- `<method>_critic_<iteration>.pth`

## Evaluation

Our training code will automatically evaluate and save to wandb, so you can just check you wandb log. Still we also provide evaluate script as shown below.

`bandit/main.py` supports direct evaluation via `--mode eval`.

Example (bandit):

```bash
cd bandit
python main.py \
  --mode eval \
  --method mepoly \
  --task lemniscate \
  --actor_model run/mepoly/mepoly_actor.pth
```

Outputs include:
- distribution plots for bandit
- trajectory visualizations for smoothworld (generated during periodic evaluation in training)

## Key Method Components

- **MePoly policy**:
  - `bandit/models/mepoly.py`
  - `smoothworld/models/mepoly.py`
- **FPO objective and rollout logic**:
  - `bandit/models/fpo.py`
  - `smoothworld/models/fpo.py`
- **Base PPO training loop**:
  - `bandit/models/ppo.py`
  - `smoothworld/models/ppo.py`
- **Environments**:
  - `bandit/utils/bandit.py`
  - `smoothworld/envs/smoothworld.py`

<!-- ## Citation

If you use this code, please cite:

```bibtex
@article{mepoly2026,
  title={MePoly: Max Entropy Polynomial Policy Optimization},
  author={MePoly Team},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
``` -->

Please update the citation entry with the final authors and publication venue when available.

## License

This project is released under the MIT License. See `LICENSE`.

## Acknowledgments

Parts of this codebase build on [FPO](https://github.com/akanazawa/fpo).
