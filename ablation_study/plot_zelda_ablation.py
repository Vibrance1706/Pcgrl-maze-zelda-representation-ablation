import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch


# STYLE

sns.set_theme(style="whitegrid", context="notebook")
sns.set_palette("tab10")

plt.rcParams.update({
    "figure.figsize": (7, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

os.makedirs("outputs/plots", exist_ok=True)


# LOAD DATA

df = pd.read_csv("outputs/zelda_ablation_mod.csv")
df["training_steps"] = df["training_steps"].astype(int)

df_base = df[df["ablation"] == "baseline"]


# 1. LEARNING CURVES

g = sns.FacetGrid(
    df[df["entropy"] == 0.01],
    col="representation",
    hue="ablation",
    height=4,
    aspect=1
)

g.map_dataframe(
    sns.lineplot,
    x="training_steps",
    y="success_rate",
    marker="o"
)

g.add_legend(title="Ablation")

if g._legend is not None:
    leg = g._legend
    leg.set_bbox_to_anchor((1.02, 1))   # move slightly outside
    leg.set_loc("upper right")
    leg.set_frame_on(False)

plt.tight_layout(pad=1.2)

plt.savefig(
    "outputs/plots/learning_curves.png",
    dpi=300,
    bbox_inches="tight"   # prevents clipping
)
plt.close()


# 2. ENTROPY EFFECT

g = sns.FacetGrid(
    df_base,
    col="representation",
    hue="entropy",
    height=4
)

g.map_dataframe(
    sns.lineplot,
    x="training_steps",
    y="success_rate",
    marker="o"
)

g.add_legend(title="Entropy")

if g._legend is not None:
    leg = g._legend
    leg.set_bbox_to_anchor((0.98, 0.02))
    leg.set_loc("upper right")
    leg.set_frame_on(False)
    for text in leg.get_texts():
        text.set_fontsize(9)

plt.tight_layout(pad=1.2)
plt.savefig("outputs/plots/entropy_effect.png", dpi=300)
plt.close()


# 3. REPRESENTATION COMPARISON

plt.figure()

sns.lineplot(
    data=df_base[df_base["entropy"] == 0.01],
    x="training_steps",
    y="success_rate",
    hue="representation",
    marker="o",
    linewidth=2
)

plt.legend(loc="upper right", frameon=False, fontsize=9)

plt.title("Representation Impact")
plt.tight_layout(pad=1.2)

plt.savefig("outputs/plots/representation.png", dpi=300)
plt.close()


# 4. BARPLOTS (FIXED)

g = sns.catplot(
    data=df,
    kind="bar",
    x="representation",
    y="success_rate",
    hue="ablation",
    col="training_steps",
    col_wrap=3,
    height=4,
    aspect=1,
    errorbar=None,
    legend=False
)

g.set_titles("{col_name} Steps")
g.set_axis_labels("", "Success Rate")

# Single clean legend
ablations = df["ablation"].unique()

# Use same palette as seaborn
palette = sns.color_palette("tab10", len(ablations))

# Create legend handles manually
handles = [
    Patch(facecolor=palette[i], label=ablations[i])
    for i in range(len(ablations))
]

# Add legend
plt.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=len(ablations),
    frameon=False,
    fontsize=10
)

# plt.tight_layout(pad=1.2)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("outputs/plots/barplots.png", dpi=300)
plt.close()


# 5. REWARD vs SUCCESS

plt.figure()

sns.scatterplot(
    data=df,
    x="avg_reward",
    y="success_rate",
    hue="representation",
    style="ablation",
    s=80,
    alpha=0.8
)

plt.legend(loc="lower right", frameon=False, fontsize=9)

plt.title("Reward vs Success")
plt.tight_layout(pad=1.2)

plt.savefig("outputs/plots/reward_vs_success.png", dpi=300)
plt.close()


# 6. STABILITY

plt.figure()

sns.lineplot(
    data=df_base[df_base["entropy"] == 0.01],
    x="training_steps",
    y="std_success",
    hue="representation",
    marker="o"
)

plt.legend(loc="lower right", frameon=False, fontsize=9)

plt.title("Training Stability")
plt.tight_layout(pad=1.2)

plt.savefig("outputs/plots/stability.png", dpi=300)
plt.close()


# 7. HEATMAPS

g = sns.FacetGrid(
    df[df["ablation"] == "baseline"],
    col="training_steps",
    height=4
)

def heatmap(data, **kwargs):
    pivot = data.pivot_table(
        values="success_rate",
        index="representation",
        columns="entropy"
    )
    sns.heatmap(pivot, annot=True, fmt=".2f", cbar=False)

g.map_dataframe(heatmap)

plt.tight_layout(pad=1.2)
plt.savefig("outputs/plots/heatmaps.png", dpi=300)
plt.close()


# 8. PATH vs SUCCESS

plt.figure()

sns.scatterplot(
    data=df,
    x="avg_path_length",
    y="success_rate",
    hue="representation",
    style="ablation",
    s=80
)

plt.legend(loc="upper right", frameon=False, fontsize=9)

plt.title("Path Quality vs Success")
plt.tight_layout(pad=1.2)

plt.savefig("outputs/plots/path_vs_success.png", dpi=300)
plt.close()


# 9. EMPTY RATIO

plt.figure()

sns.lineplot(
    data=df_base,
    x="training_steps",
    y="avg_empty_ratio",
    hue="representation",
    style="entropy",
    marker="o"
)

plt.legend(loc="upper right", frameon=False, fontsize=9)

plt.title("Empty Ratio (Density)")
plt.tight_layout(pad=1.2)

plt.savefig("outputs/plots/empty_ratio.png", dpi=300)
plt.close()