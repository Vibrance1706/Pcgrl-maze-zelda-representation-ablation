import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import ast
from matplotlib.lines import Line2D


# LOAD DATA

df = pd.read_csv("outputs/experiment_results.csv") # Change for Zelda, change the file names too for the graphs

df.columns = df.columns.str.strip()

if "atd_reward" in df.columns:
    df.rename(columns={"atd_reward": "std_reward"}, inplace=True)

df["seed_rewards"] = df["seed_rewards"].apply(ast.literal_eval)
df["seed_success"] = df["seed_success"].apply(ast.literal_eval)

os.makedirs("outputs/plots", exist_ok=True)

narrow = df[df["representation"] == "narrow"].sort_values("training_steps")
wide = df[df["representation"] == "wide"].sort_values("training_steps")

# Styling
plt.rcParams["font.size"] = 11

# Seeds
SEEDS = list(range(10))
cmap = plt.get_cmap("tab10")
seed_colors = {seed: cmap(seed) for seed in SEEDS}



def add_legends_scatter():
    # Seed legend
    seed_handles = [
        Line2D([0], [0], marker='o', color=seed_colors[s], linestyle='None', label=f"Seed {s}")
        for s in SEEDS
    ]

    seed_legend = plt.legend(
        handles=seed_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.7),
        title="Seeds",
        fontsize=8
    )

    # Representation legend
    rep_handles = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', label='Narrow'),
        Line2D([0], [0], marker='x', color='black', linestyle='None', label='Wide')
    ]

    plt.legend(
        handles=rep_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.3),
        title="Representation"
    )

    plt.gca().add_artist(seed_legend)


def add_legends_lines():
    # Seed legend
    seed_handles = [
        Line2D([0], [0], color=seed_colors[s], linestyle='-', label=f"Seed {s}")
        for s in SEEDS
    ]

    seed_legend = plt.legend(
        handles=seed_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.7),
        title="Seeds",
        fontsize=8
    )

    # Representation legend
    rep_handles = [
        Line2D([0], [0], color='black', linestyle='-', marker='o', label='Narrow'),
        Line2D([0], [0], color='black', linestyle='--', marker='x', label='Wide')
    ]

    plt.legend(
        handles=rep_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.3),
        title="Representation"
    )

    plt.gca().add_artist(seed_legend)



# 1. SUCCESS RATE

plt.figure(figsize=(9, 5.5))

plt.plot(narrow["training_steps"], narrow["success_rate"], marker="o", linewidth=2.5)
plt.plot(wide["training_steps"], wide["success_rate"], marker="x", linestyle="--", linewidth=2.5)

plt.title("Success Rate vs Training Steps")
plt.xlabel("Training Steps")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.grid(alpha=0.3)

# Simple legend (no seeds here)
plt.legend(["Narrow", "Wide"], loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("outputs/plots/success_rate_basic_binary.png", dpi=300)
plt.show()



# 2. REWARD vs SUCCESS

plt.figure(figsize=(9, 5.5))

plt.scatter(narrow["avg_reward"], narrow["success_rate"], s=90, marker="o")
plt.scatter(wide["avg_reward"], wide["success_rate"], s=90, marker="x")

for i, step in enumerate(narrow["training_steps"]):
    plt.annotate(f"N-{int(step/1000)}k", (narrow["avg_reward"].iloc[i], narrow["success_rate"].iloc[i]))

for i, step in enumerate(wide["training_steps"]):
    plt.annotate(f"W-{int(step/1000)}k", (wide["avg_reward"].iloc[i], wide["success_rate"].iloc[i]))

plt.title("Reward vs Success Alignment")
plt.xlabel("Average Reward")
plt.ylabel("Success Rate")
plt.grid(alpha=0.3)

plt.legend(["Narrow", "Wide"], loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("outputs/plots/reward_vs_success_binary.png", dpi=300)
plt.show()


# 3. SEED REWARD DISTRIBUTION

plt.figure(figsize=(9, 5.5))

steps = narrow["training_steps"].values
positions = np.arange(len(steps))
width = 0.25

for i, step in enumerate(steps):
    rewards_n = narrow["seed_rewards"].iloc[i]
    rewards_w = wide["seed_rewards"].iloc[i]

    for j, seed in enumerate(SEEDS):
        plt.scatter(positions[i] - width, rewards_n[j], color=seed_colors[seed], marker="o", s=70)
        plt.scatter(positions[i] + width, rewards_w[j], color=seed_colors[seed], marker="x", s=70)

plt.xticks(positions, steps)
plt.title("Seed-Level Reward Distribution")
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.grid(alpha=0.3)

add_legends_scatter()

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("outputs/plots/reward_distribution_seeds_binary.png", dpi=300)
plt.show()


# 4. REWARD PROGRESSION

plt.figure(figsize=(9, 5.5))

plt.plot(narrow["training_steps"], narrow["avg_reward"], marker="o", linewidth=2.5)
plt.plot(wide["training_steps"], wide["avg_reward"], marker="x", linestyle="--", linewidth=2.5)

plt.fill_between(
    narrow["training_steps"],
    narrow["avg_reward"] - narrow["std_reward"],
    narrow["avg_reward"] + narrow["std_reward"],
    alpha=0.2
)

plt.fill_between(
    wide["training_steps"],
    wide["avg_reward"] - wide["std_reward"],
    wide["avg_reward"] + wide["std_reward"],
    alpha=0.2
)

plt.title("Reward vs Training Steps")
plt.xlabel("Training Steps")
plt.ylabel("Average Reward")
plt.grid(alpha=0.3)

plt.legend(["Narrow", "Wide"], loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("outputs/plots/reward_progression_binary.png", dpi=300)
plt.show()


# 5. SEED SUCCESS TRAJECTORIES

plt.figure(figsize=(9, 5.5))

steps = narrow["training_steps"].values

narrow_success = np.array(narrow["seed_success"].tolist())
wide_success = np.array(wide["seed_success"].tolist())

for i, seed in enumerate(SEEDS):

    plt.plot(
        steps,
        narrow_success[:, i],
        color=seed_colors[seed],
        linestyle="-",
        linewidth=1.5,        
        marker="o",
        markersize=5,
        markevery=2,            
        alpha=0.8,              
        zorder=2
    )

    # Wide (dashed)
    plt.plot(
        steps,
        wide_success[:, i],
        color=seed_colors[seed],
        linestyle="--",
        linewidth=1.5,
        marker="x",
        markersize=5,
        markevery=2,
        alpha=0.8,
        zorder=1
    )

plt.title("Seed-wise Success Rate Comparison")
plt.xlabel("Training Steps")
plt.ylabel("Success Rate")
plt.ylim(0, 1)

plt.grid(alpha=0.3)

add_legends_lines()

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("outputs/plots/seed_success_trajectories_binary.png", dpi=300)
plt.show()