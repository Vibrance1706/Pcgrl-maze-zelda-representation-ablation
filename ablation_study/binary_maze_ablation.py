import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
import torch
import csv
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.binary_env import BinaryPCGRLEnv
from env.utils import shortest_path_start_goal
from env.problem import connectivity_ratio

# CONFIG
TRAINING_STEPS_LIST = [500000, 1000000]
REPRESENTATIONS = ["narrow", "wide"]
SEEDS = [0, 42]
EPISODES = 100

CSV_PATH = "outputs/binary_maze_ablation.csv"


# ABLATIONS

ABLATIONS = [
    {"name": "baseline", "config": {}},
    {"name": "no_connectivity", "config": {"use_connectivity": False}},
    {"name": "no_corridor", "config": {"use_corridor": False}},
]

# Entropy ablation
ENTROPY_VALUES = [0.01, 0.05]


# SEED

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# TRAIN

def train_model(rep, steps, seed, problem_config, ent_coef):
    set_seed(seed)

    env = make_vec_env(
        lambda: BinaryPCGRLEnv(
            representation=rep,
            problem_config=problem_config
        ),
        n_envs=4,
        seed=seed
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        ent_coef=ent_coef,
        seed=seed
    )

    model.learn(total_timesteps=steps)
    return model


# EVALUATE

def evaluate_model(model, rep, seed, problem_config):
    rewards = []
    success = 0
    path_lengths = []
    connectivities = []
    empty_ratios = []

    for _ in range(EPISODES):
        env = BinaryPCGRLEnv(
            representation=rep,
            problem_config=problem_config
        )
        obs, _ = env.reset(seed=seed)

        total_reward = 0

        for _ in range(200):
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)

        # SUCCESS

        path_len = shortest_path_start_goal(env.grid)

        if path_len >= 15:
            success += 1

        # EXTRA METRICS

        if path_len > 0:
            path_lengths.append(path_len)

        connectivities.append(connectivity_ratio(env.grid))
        empty_ratios.append(np.mean(env.grid == 0))

    return (
        np.mean(rewards),
        success / EPISODES,
        np.mean(path_lengths) if path_lengths else 0,
        np.mean(connectivities) if connectivities else 0,
        np.mean(empty_ratios) if empty_ratios else 0
    )

# LOGGING

def log(row):
    os.makedirs("outputs", exist_ok=True)
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "ablation",
                "entropy",
                "representation",
                "training_steps",
                "avg_reward",
                "std_reward",
                "success_rate",
                "std_success",
                "avg_path_length",
                "avg_connectivity",
                "avg_empty_ratio",
                "seed_rewards",
                "seed_success",
                "seed_paths"
            ])

        writer.writerow(row)


# MAIN

def run_ablation():
    start = time.time()

    for ablation in ABLATIONS:
        ablation_name = ablation["name"]
        config = ablation["config"]

        for ent in ENTROPY_VALUES:
            for steps in TRAINING_STEPS_LIST:
                for rep in REPRESENTATIONS:

                    seed_rewards = []
                    seed_success = []
                    seed_paths = []
                    seed_conns = []
                    seed_density = []

                    for seed in SEEDS:
                        model = train_model(rep, steps, seed, config, ent)

                        avg_reward, success_rate, avg_path, avg_conn, avg_density = evaluate_model(
                            model, rep, seed, config
                        )

                        seed_rewards.append(avg_reward)
                        seed_success.append(success_rate)
                        seed_paths.append(avg_path)
                        seed_conns.append(avg_conn)
                        seed_density.append(avg_density)

                    log([
                        ablation_name,
                        ent,
                        rep,
                        steps,
                        np.mean(seed_rewards),
                        np.std(seed_rewards),
                        np.mean(seed_success),
                        np.std(seed_success),
                        np.mean(seed_paths),
                        np.mean(seed_conns),
                        np.mean(seed_density),
                        seed_rewards,
                        seed_success,
                        seed_paths
                    ])

                    print(f"{ablation_name} | ent={ent} | {rep} | {steps} done")

    print("Total time:", time.time() - start)


if __name__ == "__main__":
    run_ablation()