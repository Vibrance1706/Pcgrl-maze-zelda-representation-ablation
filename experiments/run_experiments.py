import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.binary_env import BinaryPCGRLEnv
from env.utils import is_connected, shortest_path_length
import random
import torch
import time

# CONFIGURATION
TRAINING_STEPS_LIST = [50000, 100000, 200000, 500000, 750000, 1000000]
REPRESENTATIONS = ["narrow", "wide"]
EPISODES = 500
SEEDS = [0, 1, 2, 3, 4 ,5, 6, 7, 8, 9]

CSV_PATH = "outputs/experiment_results.csv"



# SEED FUNCTION

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



# TRAIN FUNCTION

def train_model(representation, steps, seed):
    print(f"\nTraining {representation} for {steps} steps | seed={seed}")

    set_seed(seed)
    start_time = time.time()

    env = make_vec_env(
        lambda: BinaryPCGRLEnv(representation=representation),
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
        seed=seed,
        ent_coef=0.03
    )

    model.learn(total_timesteps=steps)

    end_time = time.time()
    print(f" Training time: {end_time - start_time:.2f} seconds")

    os.makedirs("outputs/models", exist_ok=True)
    model.save(f"outputs/models/{representation}_{steps}_seed{seed}.zip")

    return model



# BFS PATH (UNWEIGHTED)

def bfs_path(grid, start, goal):
    from collections import deque

    rows, cols = grid.shape
    queue = deque([(start, [start])])
    visited = set([start])

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == goal:
            return path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

    return []


# EVALUATION FUNCTION

def evaluate_model(model, representation, seed):
    rewards = []
    success = 0

    for _ in range(EPISODES):
        env = BinaryPCGRLEnv(representation=representation)
        obs, _ = env.reset(seed=seed)

        total_reward = 0

        for _ in range(100):
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        START = (0,0)
        GOAL = (env.grid.shape[0] - 1, env.grid.shape[1] - 1)
        path = bfs_path(env.grid, START, GOAL)
        if len(path) >= 15:
            success += 1

    return np.mean(rewards), success / EPISODES



# VISUALIZATION

def visualize_best_model(model, representation, steps, seed, trials=20):
    best_grid = None
    best_score = -float("inf")

    for _ in range(trials):
        env = BinaryPCGRLEnv(representation=representation)
        obs, _ = env.reset(seed=seed)

        total_reward = 0

        for _ in range(100):
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break

        path_len = shortest_path_length(env.grid)
        connected = is_connected(env.grid)

        score = total_reward
        if connected:
            score += 50
        if path_len > 0:
            score += path_len

        if score > best_score:
            best_score = score
            best_grid = env.grid.copy()


    # FIXED START & GOAL

    START = (0, 0)
    GOAL = (best_grid.shape[0] - 1, best_grid.shape[1] - 1)

    # DISPLAY GRID

    display_grid = 1 - best_grid
    plt.imshow(display_grid, cmap="gray")

    # DRAW PATH

    path = bfs_path(best_grid, START, GOAL)

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        plt.plot(xs, ys, color='blue', linewidth=2)

    # DRAW START & GOAL

    plt.scatter(START[1], START[0], c='green', s=80)
    plt.scatter(GOAL[1], GOAL[0], c='red', s=80)

    # SAVE

    os.makedirs("outputs/plots", exist_ok=True)
    filename = f"{'ngg' if representation=='narrow' else 'wgg'}_{steps}.png"
    plt.title(f"{representation} | {steps} | BEST")
    plt.axis("off")
    plt.savefig(f"outputs/plots/{filename}")
    plt.close()

# CSV LOGGER

def log_results(row, seed_rewards, seed_success):
    os.makedirs("outputs", exist_ok=True)
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "representation",
                "training_steps",
                "avg_reward",
                "std_reward",
                "success_rate",
                "std_success",
                "seed_rewards",
                "seed_success"
            ])

        writer.writerow(row + [seed_rewards, seed_success])


# MAIN LOOP

def run_experiments(show_grids=False):

    total_start = time.time()

    for steps in TRAINING_STEPS_LIST:
        for rep in REPRESENTATIONS:

            seed_rewards = []
            seed_success = []
            
            best_model = None
            best_score = -float("inf")
            best_seed = None

            for seed in SEEDS:
                model = train_model(rep, steps, seed)

                avg_reward, success_rate = evaluate_model(model, rep, seed)

                seed_rewards.append(avg_reward)
                seed_success.append(success_rate)
                score = success_rate
            
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_seed = seed

            mean_reward = np.mean(seed_rewards)
            std_reward = np.std(seed_rewards)

            mean_success = np.mean(seed_success)
            std_success = np.std(seed_success)

            print(f"\nFINAL RESULT → {rep} | {steps}")
            print(f"Avg reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"Success rate: {mean_success:.3f} ± {std_success:.3f}")

            log_results(
                [rep, steps, mean_reward, std_reward, mean_success, std_success],
                seed_rewards,
                seed_success
            )

            if show_grids:
                visualize_best_model(best_model, rep, steps, best_seed)
                
                
    print(f"Best seed for {rep}-{steps}: {best_seed}, Score: {best_score:.3f}")
    total_end = time.time()
    print(f"\nTotal experiment time: {total_end - total_start:.2f} seconds")


if __name__ == "__main__":
    run_experiments(show_grids=True)