"""
This code reads training log file and generates HOTA Curves graph as pdf and esp. standard format for Q1 journals
HOTA Curves: (DetA vs. AssA): Plot Detection Accuracy ($DetA$) and Association Accuracy ($AssA$) against localization thresholds. This specific graph requires the full HOTA breakdown to see whether your tracker's main bottleneck is object detection or frame-to-frame identity matching.
How to run: python src/generate_hota_curves.py

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION (Change these paths or experiment variants dynamically) ---
MODEL_GRAPH_DIR = "results/egohumans_appearance2_graphs"
os.makedirs(MODEL_GRAPH_DIR, exist_ok=True)

# Dictionary pointing to your respective experiment evaluation outputs
# Key: Plot Display Label | Value: Target CSV location path
"""
TARGET_SETUPS = {
    "TrackEgoFormer (With ReID)": "results/TrackEgoFormerAppearance_ReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormerAppearance_ReID_TEST/pedestrian_detailed.csv",
    "TrackEgoFormer (No ReID)": "results/TrackEgoFormerAppearance_NoReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormerAppearance_NoReID_TEST/pedestrian_detailed.csv",
    "TrackEgoFormer (Best MOTA Variant)": "results/TrackEgoFormerAppearance_ReID_best_MOTA_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormerAppearance_ReID_best_MOTA_TEST/pedestrian_detailed.csv"
}
"""
TARGET_SETUPS = {
    "TrackEgoFormer (With ReID)": "results/TrackEgoFormerAppearance_ver2_ReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormerAppearance_ver2_ReID_TEST/pedestrian_detailed.csv",
    "TrackEgoFormer (No ReID)": "results/TrackEgoFormerAppearance_ver2_NoReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormerAppearance_ver2_NoReID_TEST/pedestrian_detailed.csv",
    "TrackEgoFormer (Best MOTA Variant)": "results/TrackEgoFormerAppearance_ReID_best_MOTA_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormerAppearance_ReID_best_MOTA_TEST/pedestrian_detailed.csv"
}

# Publication-quality styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "grid.alpha": 0.3
})

# Localization thresholds evaluated by TrackEval framework (0.05 to 0.95 with steps of 0.05)
thresholds = np.arange(0.05, 1.00, 0.05) 
csv_columns_det = [f"DetA___{int(t*100)}" for t in thresholds]
csv_columns_ass = [f"AssA___{int(t*100)}" for t in thresholds]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
colors = ["#d62728", "#1f77b4", "#2ca02c"]

for (label, csv_path), color in zip(TARGET_SETUPS.items(), colors):
    if not os.path.exists(csv_path):
        print(f"[!] Warning: Target CSV missing at path: {csv_path}. Skipping this variant line.")
        continue
        
    df = pd.read_csv(csv_path)
    
    # Filter out individual sequence slices to isolate the cumulative performance summary row
    df_combined = df[df["seq"].str.lower() == "combined"]
    if df_combined.empty:
        df_combined = df.iloc[[-1]] # Fallback to final summary row if name mismatch occurs
        
    # Extract data arrays
    det_curves = df_combined[csv_columns_det].values.flatten()
    ass_curves = df_combined[csv_columns_ass].values.flatten()
    
    # Plot Detection Accuracy (DetA)
    ax1.plot(thresholds, det_curves, color=color, linewidth=2, marker="o", markersize=4, label=label)
    
    # Plot Association Accuracy (AssA)
    ax2.plot(thresholds, ass_curves, color=color, linewidth=2, marker="x", markersize=5, linestyle="--", label=label)

# Subplot 1: Detection Localization Curves
ax1.set_xlabel("Localization Threshold Alpha ") # ($\alpha$)")
ax1.set_ylabel("Accuracy Score")
ax1.set_title("Detection Accuracy ($DetA$ Curve)", pad=10)
ax1.set_xlim(0, 1)
ax1.grid(True, linestyle="--")
ax1.legend()

# Subplot 2: Identity Association Curves
ax2.set_xlabel("Localization Threshold Alpha ") # ($\alpha$)")
ax2.set_title("Association Accuracy ($AssA$ Curve)", pad=10)
ax2.set_xlim(0, 1)
ax2.grid(True, linestyle="--")
ax2.legend()

plt.tight_layout()
fig.savefig(os.path.join(MODEL_GRAPH_DIR, "hota_curves_breakdown.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(MODEL_GRAPH_DIR, "hota_curves_breakdown.eps"), bbox_inches="tight")
fig.savefig(os.path.join(MODEL_GRAPH_DIR, "hota_curves_breakdown.png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"[+] Graph 4 diagnostic plot successfully saved to {MODEL_GRAPH_DIR}")
