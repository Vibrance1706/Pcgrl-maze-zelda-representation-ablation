import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
import torch
import csv
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.zelda_env import ZeldaPCGRLEnv
from env.zelda_problem import ZeldaProblem, find_tile, bfs, PLAYER, KEY, DOOR, EMPTY



TRAINING_STEPS_LIST = [50000, 100000, 200000]
REPRESENTATIONS = ["narrow", "wide"]
SEEDS = [0, 42]
EPISODES = 100

CSV_PATH = "outputs/zelda_ablation_mod.csv"



ABLATIONS = [
    {"name": "baseline", "config": {}},
    {"name": "no_sequential", "config": {"use_sequential": False}},
    {"name": "no_distance", "config": {"use_distance_reward": False}},
    {"name": "no_connectivity", "config": {"enforce_connectivity": False}},
]

ENTROPY_VALUES = [0.01, 0.05]



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def train_model(rep, steps, seed, config, ent):
    set_seed(seed)

    env = make_vec_env(
        lambda: ZeldaPCGRLEnv(
            representation=rep,
            problem_config=config
        ),
        n_envs=4,
        seed=seed
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        ent_coef=ent,
        seed=seed
    )

    model.learn(total_timesteps=steps)
    return model


def evaluate_model(model, rep, seed, config):
    problem = ZeldaProblem(config=config)

    rewards = []
    successes = []
    valid_entity_counts = []

    path_lengths = []
    connectivities = []
    empty_ratios = []

    for _ in range(EPISODES):
        env = ZeldaPCGRLEnv(
            representation=rep,
            problem_config=config
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

        # -------------------------
        # SUCCESS
        # -------------------------
        success = problem.compute_success(env.grid)
        successes.append(success)

        # -------------------------
        # ENTITY VALIDITY
        # -------------------------
        players = find_tile(env.grid, PLAYER)
        keys = find_tile(env.grid, KEY)
        doors = find_tile(env.grid, DOOR)

        valid = int(len(players) == 1 and len(keys) == 1 and len(doors) == 1)
        valid_entity_counts.append(valid)

        # -------------------------
        # STRUCTURAL METRICS
        # -------------------------
        if valid:
            player = tuple(players[0])
            key = tuple(keys[0])
            door = tuple(doors[0])

            d1 = bfs(env.grid, player, key)
            d2 = bfs(env.grid, key, door)

            if d1 > 0 and d2 > 0:
                path_lengths.append(d1 + d2)

        connectivities.append(success)

        # Empty ratio
        empty_ratio = np.mean(env.grid == EMPTY)
        empty_ratios.append(empty_ratio)

    return (
        np.mean(rewards),
        np.mean(successes),
        np.mean(valid_entity_counts),
        np.mean(path_lengths) if len(path_lengths) > 0 else 0,
        np.mean(connectivities),
        np.mean(empty_ratios),
        rewards,
        successes,
        path_lengths
    )


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
                "entity_valid_ratio",
                "avg_path_length",
                "avg_connectivity",
                "avg_empty_ratio",
                "seed_rewards",
                "seed_success",
                "seed_paths"
            ])

        writer.writerow(row)



def run_ablation():
    start = time.time()

    for ablation in ABLATIONS:
        name = ablation["name"]
        config = ablation["config"]

        for ent in ENTROPY_VALUES:
            for steps in TRAINING_STEPS_LIST:
                for rep in REPRESENTATIONS:

                    seed_rewards = []
                    seed_success = []
                    seed_entities = []
                    seed_paths = []
                    seed_empty = []
                    seed_connectivity = []

                    for seed in SEEDS:
                        print(f"\nTraining → {name} | ent={ent} | {rep} | {steps} | seed={seed}")

                        model = train_model(rep, steps, seed, config, ent)

                        r, s, e, p, c, er, r_list, s_list, p_list = evaluate_model(
                            model, rep, seed, config
                        )

                        seed_rewards.append(r)
                        seed_success.append(s)
                        seed_entities.append(e)
                        seed_paths.append(p)
                        seed_connectivity.append(c)
                        seed_empty.append(er)

                    # Aggregate
                    avg_reward = np.mean(seed_rewards)
                    std_reward = np.std(seed_rewards)

                    avg_success = np.mean(seed_success)
                    std_success = np.std(seed_success)

                    avg_entities = np.mean(seed_entities)
                    avg_path = np.mean(seed_paths)
                    
                    avg_connectivity = np.mean(seed_connectivity)
                    avg_empty = np.mean(seed_empty)

                    print(f"\nRESULT → {name} | {rep} | {steps}")
                    print(f"Reward: {avg_reward:.3f} ± {std_reward:.3f}")
                    print(f"Success: {avg_success:.3f} ± {std_success:.3f}")

                    # Log
                    log([
                        name,
                        ent,
                        rep,
                        steps,
                        avg_reward,
                        std_reward,
                        avg_success,
                        std_success,
                        avg_path,
                        avg_connectivity,
                        avg_empty, 
                        seed_rewards,
                        seed_success,
                        seed_paths
                    ])

                    print(f"{name} | ent={ent} | {rep} | {steps} done")

    print("Total time:", time.time() - start)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_ablation()