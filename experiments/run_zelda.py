import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.zelda_env import ZeldaPCGRLEnv
from env.zelda_problem import ZeldaProblem, find_tile, bfs_path, PLAYER, KEY, DOOR

from matplotlib.colors import ListedColormap


# CONFIG

TRAINING_STEPS_LIST = [50000, 100000, 200000, 500000, 750000, 1000000]
REPRESENTATIONS = ["narrow", "wide"]
SEEDS = [0, 1, 2, 3, 4 ,5, 6, 7, 8, 9]
EPISODES = 500

CSV_PATH = "outputs/zelda_results.csv"



# SEED

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# TRAIN

def train_model(rep, steps, seed):
    print(f"\nTraining Zelda | {rep} | {steps} | seed={seed}")

    set_seed(seed)

    env = make_vec_env(
        lambda: ZeldaPCGRLEnv(representation=rep),
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
        ent_coef=0.03,
        seed=seed
    )

    model.learn(total_timesteps=steps)

    os.makedirs("outputs/models", exist_ok=True)
    model.save(f"outputs/models/zelda_{rep}_{steps}_seed{seed}.zip")

    return model

# EVALUATE

def evaluate_model(model, rep, seed):
    problem = ZeldaProblem()

    rewards = []
    successes = []

    for _ in range(EPISODES):
        env = ZeldaPCGRLEnv(representation=rep)
        obs, _ = env.reset(seed=seed)

        total_reward = 0

        for _ in range(200):
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)


        # SUCCESS CALCULATION

        success = problem.compute_success(env.grid)
        successes.append(success)

    return (
        np.mean(rewards),
        np.std(rewards),
        np.mean(successes),   # success_rate
        np.std(successes)     # std_success
    )


# LOGGING

def log_results(row):
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

        writer.writerow(row)


# VISUALIZATION

def show_grid(env, title, save_path=None):
    plt.imshow(env.grid, cmap="tab10")
    plt.title(title)
    plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()
    plt.close()


def visualize_model(model, rep, steps):

    env = ZeldaPCGRLEnv(representation=rep)
    obs, _ = env.reset(seed=0)

    for _ in range(200):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            break

    prefix = "ngg" if rep == "narrow" else "wgg"

    filename = f"zelda_{prefix}_step{steps}.png"
    path = f"outputs/plots/{filename}"

    os.makedirs("outputs/plots", exist_ok=True)

    # plt.imshow(env.grid, cmap="tab10")
    cmap = ListedColormap([
    "white",   
    "black",   
    "white",    
    "white",   
    "white"    
    ])

    plt.imshow(env.grid, cmap=cmap, vmin=0, vmax=4)
    plt.title(f"Zelda | {rep} | {steps}")
    plt.axis("off")

    # FIND ENTITIES

    players = find_tile(env.grid, PLAYER)
    keys = find_tile(env.grid, KEY)
    doors = find_tile(env.grid, DOOR)

# ALWAYS SHOW ENTITIES

    for p in players:
        plt.scatter(p[1], p[0], c="green", s=120, edgecolors="black")

    for k in keys:
        plt.scatter(k[1], k[0], c="orange", s=120, edgecolors="black")

    for d in doors:
        plt.scatter(d[1], d[0], c="red", s=120, edgecolors="black")

# DRAW PATHS IF POSSIBLE

    if len(players) >= 1 and len(keys) >= 1:
        path1 = bfs_path(env.grid, tuple(players[0]), tuple(keys[0]))
        if path1:
            xs, ys = zip(*path1)
            plt.plot(ys, xs, color="cyan", linewidth=2)

    if len(keys) >= 1 and len(doors) >= 1:
        path2 = bfs_path(env.grid, tuple(keys[0]), tuple(doors[0]))
        if path2:
            xs, ys = zip(*path2)
            plt.plot(ys, xs, color="yellow", linewidth=2)

    plt.savefig(path)
    plt.close()


# MAIN LOOP

def run_experiments():

    overall_start_time = time.time()
     
    for training_steps in TRAINING_STEPS_LIST:
        for representation in REPRESENTATIONS:

            seed_rewards = []
            seed_success = []

            for seed in SEEDS:
                start_time = time.time()

                model = train_model(representation, training_steps, seed)

                avg_reward, std_reward, success_rate, std_success = evaluate_model(
                    model, representation, seed
                )

                seed_rewards.append(avg_reward)
                seed_success.append(success_rate)

                seed_time = time.time() - start_time
                print(f"Seed {seed} finished in {seed_time:.2f} seconds")

            # Aggregate across seeds
            avg_reward = np.mean(seed_rewards)
            std_reward = np.std(seed_rewards)

            success_rate = np.mean(seed_success)
            std_success = np.std(seed_success)

            print(f"\nFINAL → {representation} | {training_steps}")
            print(f"Reward: {avg_reward:.3f} ± {std_reward:.3f}")
            print(f"Success: {success_rate:.3f} ± {std_success:.3f}")

            log_results([
                representation,
                training_steps,
                avg_reward,
                std_reward,
                success_rate,
                std_success,
                seed_rewards,
                seed_success
            ])

            visualize_model(model, representation, training_steps)

    total_time = time.time() - overall_start_time
    print(f"\nTotal experiment time: {total_time:.2f} seconds")

# RUN

if __name__ == "__main__":
    run_experiments()