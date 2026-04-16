# import pandas as pd
# import matplotlib.pyplot as plt
# import os


# # LOAD DATA

# CSV_PATH = "outputs/binary_maze_ablation.csv"
# df = pd.read_csv(CSV_PATH)

# os.makedirs("outputs/plots", exist_ok=True)

# def save_plot(name):
#     plt.tight_layout()
#     plt.savefig(f"outputs/plots/{name}.png")
#     plt.close()

# # 1. SUCCESS vs STEPS

# for ablation in df["ablation"].unique():
#     plt.figure()

#     subset = df[df["ablation"] == ablation]

#     for rep in ["narrow", "wide"]:
#         sub = subset[subset["representation"] == rep]

#         plt.plot(
#             sub["training_steps"],
#             sub["success_rate"],
#             marker='o',
#             label=rep
#         )

#     plt.title(f"Success vs Steps ({ablation})")
#     plt.xlabel("Training Steps")
#     plt.ylabel("Success Rate")
#     plt.legend()

#     save_plot(f"success_vs_steps_{ablation}")

# # 2. REWARD vs STEPS (WITH ERROR BARS)

# for ablation in df["ablation"].unique():
#     plt.figure()

#     subset = df[df["ablation"] == ablation]

#     for rep in ["narrow", "wide"]:
#         sub = subset[subset["representation"] == rep]

#         plt.errorbar(
#             sub["training_steps"],
#             sub["avg_reward"],
#             yerr=sub["std_reward"],
#             marker='o',
#             capsize=5,
#             label=rep
#         )

#     plt.title(f"Reward vs Steps ({ablation})")
#     plt.xlabel("Training Steps")
#     plt.ylabel("Average Reward")
#     plt.legend()

#     save_plot(f"reward_vs_steps_{ablation}")

# # 3. STABILITY (STD SUCCESS)

# plt.figure()

# for rep in ["narrow", "wide"]:
#     sub = df[(df["ablation"] == "baseline") & (df["representation"] == rep)]

#     plt.plot(
#         sub["training_steps"],
#         sub["std_success"],
#         marker='o',
#         label=rep
#     )

# plt.title("Training Stability (Std Success)")
# plt.xlabel("Training Steps")
# plt.ylabel("Std Success")
# plt.legend()

# save_plot("stability_std_success")

# # 4. CONNECTIVITY vs SUCCESS

# plt.figure()

# for rep in ["narrow", "wide"]:
#     sub = df[df["representation"] == rep]

#     plt.scatter(
#         sub["avg_connectivity"],
#         sub["success_rate"],
#         label=rep
#     )

# plt.title("Connectivity vs Success")
# plt.xlabel("Connectivity")
# plt.ylabel("Success Rate")
# plt.legend()

# save_plot("connectivity_vs_success")


# # 5. ABLATION COMPARISON

# # Choose one training step (100k)
# STEP = 100000

# subset = df[df["training_steps"] == STEP]

# plt.figure()

# x_labels = subset["ablation"].unique()
# x = range(len(x_labels))

# narrow_vals = []
# wide_vals = []

# for ab in x_labels:
#     narrow_vals.append(
#         subset[(subset["ablation"] == ab) & (subset["representation"] == "narrow")]["success_rate"].values[0]
#     )
#     wide_vals.append(
#         subset[(subset["ablation"] == ab) & (subset["representation"] == "wide")]["success_rate"].values[0]
#     )

# width = 0.35

# plt.bar([i - width/2 for i in x], narrow_vals, width=width, label="narrow")
# plt.bar([i + width/2 for i in x], wide_vals, width=width, label="wide")

# plt.xticks(x, x_labels)
# plt.title(f"Ablation Comparison (Success @ {STEP})")
# plt.ylabel("Success Rate")
# plt.legend()

# save_plot("ablation_comparison")

# # 6. ENTROPY COMPARISON

# plt.figure()

# for ent in df["entropy"].unique():
#     sub = df[(df["ablation"] == "baseline") & (df["entropy"] == ent) & (df["representation"] == "wide")]

#     plt.plot(
#         sub["training_steps"],
#         sub["success_rate"],
#         marker='o',
#         label=f"entropy={ent}"
#     )

# plt.title("Entropy Effect (Wide Representation)")
# plt.xlabel("Training Steps")
# plt.ylabel("Success Rate")
# plt.legend()

# save_plot("entropy_effect_wide")


# print("Done")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set_theme(style="whitegrid", context="notebook")
sns.set_palette("tab10")

plt.rcParams.update({
    "figure.figsize": (7, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12
})


# LOAD DATA

CSV_PATH = "outputs/binary_maze_ablation.csv"
df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = df.columns.str.strip()

# Ensure proper ordering of ablations
df["ablation"] = pd.Categorical(
    df["ablation"],
    categories=["baseline", "no_connectivity", "no_corridor"],
    ordered=True
)

os.makedirs("outputs/plots", exist_ok=True)


# 1. SUCCESS vs STEPS

g = sns.FacetGrid(
    df,
    col="ablation",
    hue="representation",
    height=4,
    aspect=1,
    sharey=True
)

g.map_dataframe(
    sns.lineplot,
    x="training_steps",
    y="success_rate",
    marker="o"
)

g.add_legend(title="Representation")

if g._legend is not None:
    leg = g._legend
    leg.set_bbox_to_anchor((1.02, 1))
    leg.set_loc("upper right")
    leg.set_frame_on(False)

g.set_axis_labels("Training Steps", "Success Rate")
g.set_titles("{col_name}")

plt.tight_layout(pad=1.2)
plt.savefig("outputs/plots/success_vs_steps_clean.png", dpi=300)
plt.close()



# 2. REWARD vs STEPS (MEAN ± STD)

g = sns.FacetGrid(
    df,
    col="ablation",
    hue="representation",
    height=4,
    aspect=1,
    sharey=False
)

g.map_dataframe(
    sns.lineplot,
    x="training_steps",
    y="avg_reward",
    marker="o"
)


for ax, ablation in zip(g.axes.flat, df["ablation"].cat.categories):
    sub = df[df["ablation"] == ablation]

    for rep in sub["representation"].unique():
        s = sub[sub["representation"] == rep].sort_values("training_steps")

        ax.fill_between(
            s["training_steps"],
            s["avg_reward"] - s["std_reward"],
            s["avg_reward"] + s["std_reward"],
            alpha=0.2
        )

g.add_legend(title="Representation")

if g._legend is not None:
    leg = g._legend
    leg.set_bbox_to_anchor((1.02, 1))
    leg.set_loc("upper right")
    leg.set_frame_on(False)

g.set_axis_labels("Training Steps", "Reward")
g.set_titles("{col_name}")

plt.tight_layout(pad=1.2)
plt.savefig("outputs/plots/reward_vs_steps_clean.png", dpi=300)
plt.close()

# 3. ENTROPY EFFECT

g = sns.FacetGrid(
    df[df["ablation"] == "baseline"],
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

plt.tight_layout(pad=1.2)
plt.savefig("outputs/plots/entropy_effect_clean.png", dpi=300)
plt.close()


# 4. CONNECTIVITY vs SUCCESS

plt.figure()

sns.scatterplot(
    data=df,
    x="avg_connectivity",
    y="success_rate",
    hue="representation",
    style="ablation",
    s=80
)

plt.legend(loc="lower right", frameon=False, fontsize=9)
plt.title("Connectivity vs Success")

plt.tight_layout(pad=1.2)
plt.savefig("outputs/plots/connectivity_vs_success_clean.png", dpi=300)
plt.close()


print("Binary ablation plots (clean style) generated.")
