# from stable_baselines3 import PPO
# from env.binary_env import BinaryPCGRLEnv
# from env.utils import is_connected
# import numpy as np


# def evaluate(representation, episodes=30):
#     model = PPO.load(f"outputs/models/{representation}_model")

#     rewards = []
#     success = 0

#     for _ in range(episodes):
#         env = BinaryPCGRLEnv(representation=representation)
#         obs, _ = env.reset()

#         total_reward = 0

#         for _ in range(100):
#             action, _ = model.predict(obs)
#             obs, reward, done, _, _ = env.step(action)
#             total_reward += reward

#             if done:
#                 break

#         rewards.append(total_reward)

#         if is_connected(env.grid):
#             success += 1

#     print(f"\n{representation.upper()} RESULTS")
#     print("Average reward:", np.mean(rewards))
#     print("Success rate:", success / episodes)


# if __name__ == "__main__":
#     evaluate("narrow")
#     evaluate("wide")

from stable_baselines3 import PPO
from env.binary_env import BinaryPCGRLEnv
from env.utils import is_connected
import numpy as np
import csv
import os


def evaluate(representation, episodes=30, csv_path="outputs/results.csv"):
    model = PPO.load(f"outputs/models/{representation}_model")

    rewards = []
    success = 0

    for _ in range(episodes):
        env = BinaryPCGRLEnv(representation=representation)
        obs, _ = env.reset()

        total_reward = 0

        for _ in range(100):
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

        if is_connected(env.grid):
            success += 1

    avg_reward = float(np.mean(rewards))
    success_rate = success / episodes

    print(f"\n{representation.upper()} RESULTS")
    print("Average reward:", avg_reward)
    print("Success rate:", success_rate)

    # ✅ Save to CSV
    os.makedirs("outputs", exist_ok=True)

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # write header only once
        if not file_exists:
            writer.writerow(["representation", "episodes", "avg_reward", "success_rate"])

        writer.writerow([representation, episodes, avg_reward, success_rate])


if __name__ == "__main__":
    evaluate("narrow")
    evaluate("wide")