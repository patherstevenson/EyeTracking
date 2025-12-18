import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Load CSV files
# ---------------------------------------------------------------------
best_df = pd.read_csv("logs/best_val_loss.csv")
summary_df = pd.read_csv("logs/summary_val_loss.csv")

# Ensure numeric sorting of betas
best_df["beta"] = best_df["beta"].astype(float)
summary_df["beta"] = summary_df["beta"].astype(float)
betas = sorted(summary_df["beta"].unique())


# =========================================================================
# FIGURE 1 — Best validation loss vs beta
# =========================================================================
plt.figure(figsize=(7, 4))
plt.plot(best_df["beta"], best_df["val_loss"], marker="o", linewidth=2)

plt.xlabel("β (Huber SmoothL1 parameter)", fontsize=12)
plt.ylabel("Best validation loss", fontsize=12)
plt.title("Best Validation Loss for Each β", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("fig_best_val_loss.png", dpi=300)
plt.close()


# =========================================================================
# FIGURE 2 — Validation curves per β (grid of subplots)
# =========================================================================
n = len(betas)
cols = 4
rows = int(np.ceil(n / cols))

plt.figure(figsize=(18, 12))

for i, beta in enumerate(betas, 1):
    sub = summary_df[summary_df["beta"] == beta]

    plt.subplot(rows, cols, i)
    plt.plot(sub["epoch"], sub["val_loss"], marker="o", markersize=3)
    plt.title(f"β = {beta}", fontsize=10)
    plt.xlabel("Epoch", fontsize=9)
    plt.ylabel("Val loss", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)

plt.suptitle("Validation Loss Curves per β", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("fig_val_curves_per_beta.png", dpi=300)
plt.close()


# =========================================================================
# FIGURE 3 — All validation curves superposed
# =========================================================================
plt.figure(figsize=(8, 5))

cmap = plt.get_cmap("tab20", len(betas))

for i, beta in enumerate(betas):
    sub = summary_df[summary_df["beta"] == beta]
    plt.plot(sub["epoch"], sub["val_loss"], label=f"β={beta}", linewidth=1.8, color=cmap(i))

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Validation loss", fontsize=12)
plt.title("Validation Loss Curves for All β", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=8, ncol=4)
plt.tight_layout()
plt.savefig("fig_all_val_curves.png", dpi=300)
plt.close()

