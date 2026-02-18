# GridWorld FPO

This repo implements **Flow Policy Optimization (FPO)** for continuous-action policy learning in custom grid-world environments.

Please see the [blog](https://flowreinforce.github.io/) and [paper](todo) for more details.

It is based on the excellent, beginner-friendly [PPO grid-world repo by Eric Yu](https://github.com/ericyangyu/PPO-for-Beginners)

You can see the **main differences with PPO** in the following parts of the code:

- `sample_action_with_info` defined in [`diffusion_policy.py`](./models/diffusion_policy.py), called in [`fpo.py`](./models/fpo.py), which samples (eps, t) pairs and pre-computes the conditional flow matching loss ($\text{ELBO}_{old}$) in the paper
- in [`fpo.py`](./models/fpo.py) the likelihood ratio is replaced by the difference of the cfm losses.


## Installation

Dependencies (Python 3.8+):

```
pip install -r requirements.txt
```

## Quick Start

### Training
```bash
# Train FPO policy  
python main.py --method fpo

# Train PPO policy
python main.py --method ppo

# Train MEM-PPO policy (max-entropy polynomial policy)
python main.py --method mepoly

```


## Visualization

<!-- The environment supports simple matplotlib-based rendering to observe agent behavior in several ways as done in the paper.
```bash
# Visualize FPO policy action distributions
python visualize.py --method fpo

# Visualize at specific state
python visualize.py --method fpo --specific_state

``` -->

To evaluate and visualize sample trajectory rollouts from fixed states:
```bash
# Evaluate and visualize in one step (recommended)
python eval_and_visualize_trajectories.py --method fpo --actor_model run/fpo/fpo_actor_10.pth   --start-mode paper --grid-mode two_walls --episodes 50

python eval_and_visualize_trajectories.py --method mepoly --actor_model run/mepoly/mepoly_actor_10.pth  --start-mode paper --grid-mode two_walls --episodes 50

# Or just visualize existing trajectory data (evaluation always saves .pkl file)
python eval_and_visualize_trajectories.py --method ppo --actor_model ppo_actor.pth   --start-mode paper --grid-mode two_walls --episodes 50
``` 


## Grid World Configuration

Grid environments support multiple modes defined in gridworld.py:

- Configurable grid size, walls, death zones, goal zones
- Various pre-defined modes accessible via grid_mode hyperparameter

## Core Components

**Entry Point**: `main.py` - Handles training/testing modes, model loading, hyperparameter configuration

**Policy Implementations**:
- `models/ppo.py` - Base PPO algorithm implementation
- `models/fpo.py` - Flow Policy Optimization extending PPO base class
- `models/mepoly.py` - Max-entropy PPO with polynomial action distribution
- `models/diffusion_policy.py` - DiffusionPolicy class extending FeedForwardNN for flow-based sampling
- `models/network.py` - Base FeedForwardNN neural network implementation

**Environment**: `utils/gridworld.py` - Custom grid-world environment with configurable modes (two_walls, three_goals, etc.)

**Tuning hyperparameters**: The default values (shared PPO/FPO settings plus MEM-PPO specifics like `poly_order`, `poly_grid_size`, `lambda_clip`, `action_eps`) live in `main.py` inside the `hyperparameters` dict. Adjust them there before running, or change the CLI defaults in `utils/arguments.py` if you want different per-run defaults.


## Acknowledgment

Thanks again to [PPO grid-world repo by Eric Yu](https://github.com/ericyangyu/PPO-for-Beginners)!! 


