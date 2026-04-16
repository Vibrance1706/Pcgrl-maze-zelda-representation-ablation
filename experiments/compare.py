import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("outputs/zelda_results.csv")

# Clean columns (safety)
df.columns = df.columns.str.strip()

# Split by representation
narrow = df[df["representation"] == "narrow"].sort_values("training_steps")
wide = df[df["representation"] == "wide"].sort_values("training_steps")

# Create plot
plt.figure(figsize=(9, 5.5))

# ---- NARROW ----
plt.plot(
    narrow["training_steps"],
    narrow["avg_reward"],
    marker="o",
    linewidth=2,
    label="Narrow"
)

plt.fill_between(
    narrow["training_steps"],
    narrow["avg_reward"] - narrow["std_reward"],
    narrow["avg_reward"] + narrow["std_reward"],
    alpha=0.25
)

# ---- WIDE ----
plt.plot(
    wide["training_steps"],
    wide["avg_reward"],
    marker="x",
    linestyle="--",
    linewidth=2,
    label="Wide"
)

plt.fill_between(
    wide["training_steps"],
    wide["avg_reward"] - wide["std_reward"],
    wide["avg_reward"] + wide["std_reward"],
    alpha=0.25
)

# Labels & styling
plt.title("Mean ± Standard Deviation (Reward)")
plt.xlabel("Training Steps")
plt.ylabel("Average Reward")
plt.grid(alpha=0.3)

# Legend outside
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# Save
os.makedirs("outputs/plots", exist_ok=True)
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("outputs/plots/mean_std_area_zelda.png", dpi=300)

plt.show()