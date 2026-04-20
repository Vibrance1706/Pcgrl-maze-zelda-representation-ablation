# Procedural Content Generation via Reinforcement Learning (PCGRL)

This project implements and evaluates **Procedural Content Generation via Reinforcement Learning (PCGRL)** using Proximal Policy Optimization (PPO). The system generates structured environments such as **binary mazes** and **Zelda-like levels**, and studies the impact of **representation, reward design, and training configurations** through extensive experiments and ablation studies.

---

## Overview

The core idea is to train an RL agent to iteratively construct grid-based environments. The agent modifies tiles in a grid to optimize structural properties such as connectivity, path length, and playability.

Two environments are implemented:

* **Binary Maze Environment**

  * Grid of 0 (empty) and 1 (wall)
  * Goal: Generate connected mazes with long valid paths

* **Zelda Environment**

  * Multi-tile grid with:

    * Start
    * Key 
    * Goal
    * Walls / Empty space
  * Goal: Ensure solvable sequence: **Start → Key → Goal**

---

## Key Features

* Two action representations:

  * **Narrow Representation** (sequential tile updates)
  * **Wide Representation** (direct tile selection)

* Custom reward functions incorporating:

  * Connectivity
  * Path length
  * Density control
  * Corridor structure
  * Local smoothness
  * Exploration bonus

* PPO-based training using Stable-Baselines3

* Multi-seed evaluation for robustness

* Extensive visualization and plotting utilities

* Ablation studies:

  * Reward component removal
  * Entropy variation
  * Structural constraints

---

## Project Structure

```
├── env/
│   ├── binary_env.py
│   ├── zelda_env.py
│   ├── problem.py
│   ├── zelda_problem.py
│   ├── representation.py
│   ├── utils.py
│
├── experiments/
│   ├── run_experiments.py
│   ├── run_zelda.py
│   ├── plot_results.py
│
├── ablation_study/
│   ├── binary_maze_ablation.py
│   ├── zelda_ablation_study.py
│   ├── plot_binary_ablation.py
│   ├── plot_zelda_ablation.py
│
├── outputs/
│   ├── models/
│   ├── plots/
│   ├── *.csv
│
└── README.md
```

---

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

### Core Dependencies

* Python 3.10+
* gymnasium
* stable-baselines3
* numpy
* pandas
* matplotlib
* seaborn
* torch

---

## How to Run

### 1. Binary Maze Experiments

```bash
python experiments/run_experiments.py
```

Outputs:

* `outputs/experiment_results.csv`
* trained models in `outputs/models/`
* visualizations in `outputs/plots/`

---

### 2. Zelda Experiments

```bash
python experiments/run_zelda.py
```

Outputs:

* `outputs/zelda_results.csv`
* generated level visualizations

---

### 3. Binary Ablation Study

```bash
python ablation_study/binary_maze_ablation.py
```

Then plot:

```bash
python ablation_study/plot_binary_ablation.py
```

---

### 4. Zelda Ablation Study

```bash
python ablation_study/zelda_ablation_study.py
```

Then plot:

```bash
python ablation_study/plot_zelda_ablation.py
```

---

## Representations

### Narrow Representation

* Modifies one tile at a time in a fixed sequence
* Smaller action space
* Easier learning but slower adaptation

### Wide Representation

* Directly selects any tile + value
* Larger action space
* Faster but harder to learn

---

## Reward Design

### Binary Environment

* Connectivity ratio
* Path length maximization
* Density regularization
* Corridor encouragement
* Local smoothness
* Exploration bonus

### Zelda Environment

* Valid entity placement (Start, Key, Goal)
* Reachability (Start → Key → Goal)
* Distance-based shaping
* Tile diversity
* Density control

---

## Evaluation Metrics

* **Average Reward**
* **Success Rate**

  * Binary: path length ≥ threshold
  * Zelda: valid Start → Key → Goal path
* **Path Length**
* **Connectivity Ratio**
* **Density (empty tile ratio)**

---

## Key Results (Summary)

### Binary Environment

* Narrow representation consistently outperforms wide
* Connectivity reward is critical for success
* Performance saturates after ~200k–500k steps

### Zelda Environment

* Wide representation eventually surpasses narrow
* Sequential constraints significantly impact learning
* Distance-based rewards improve convergence

### Ablation Insights

* Removing connectivity severely degrades performance
* Corridor reward has moderate impact
* Entropy strongly affects exploration vs stability trade-off

---

## Visualization

The project includes:

* Learning curves (reward, success rate)
* Seed-wise variability plots
* Mean ± standard deviation area plots
* Ablation comparison plots
* Generated level visualizations

---

## Reproducibility

* Multiple seeds used for all experiments
* Deterministic seeding via NumPy, Torch, and Python RNG
* Results stored in CSV format for transparency

---

## Future Work

* Replace standard deviation with confidence intervals
* Extend to 3D or larger grid environments
* Incorporate curriculum learning
* Explore transformer-based policies
* Add human-in-the-loop evaluation

---

## References

* Khalifa et al., *PCGRL: Procedural Content Generation via Reinforcement Learning*
* Schulman et al., *Proximal Policy Optimization Algorithms*

---

## Author

Vishal M.

---
