import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# --- CONFIGURATION (Change these to generate graphs for other models) ---
# ==============================================================================
MODEL_BASE_NAME = "TrackEgoFormerEgomotion"  
OUT_DIR = "egohumans_egomotion"
# Directory where the finalized publication plots will be stored
#OUTPUT_GRAPH_DIR = f"results/{MODEL_BASE_NAME}_diagnostic_graphs"
OUTPUT_GRAPH_DIR = f"results/{OUT_DIR}_graphs"
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

# Standard root directory for your TrackEval outputs
RESULTS_ROOT = "results"
TRACKER_SUBPATH = "trackers/mot_challenge/Egohumans_full_MOT-TEST"

# 3 Distinct Colors for your 3 Checkpoints (Consistent across rows)
COLOR_MOTA = "#d62728"   # Crimson Red
COLOR_IDF1 = "#1f77b4"   # Royal Blue
COLOR_FINAL = "#2ca02c"  # Emerald Green

# Define the dataset structure explicitly mapped onto a 2x2 grid layout
# Layout coordinates: (row, col) where row 0 = With ReID, row 1 = No ReID
TARGET_SETUPS = {
    # --- ROW 0: WITH REID MODELS ---
    "Best MOTA (With ReID)":   {"folder": f"{MODEL_BASE_NAME}_ReID_best_MOTA_TEST", "color": COLOR_MOTA,  "row": 0},
    "Best IDF1 (With ReID)":   {"folder": f"{MODEL_BASE_NAME}_ReID_best_IDF1_TEST", "color": COLOR_IDF1,  "row": 0},
    "Final Epoch (With ReID)": {"folder": f"{MODEL_BASE_NAME}_ReID_TEST",           "color": COLOR_FINAL, "row": 0},
    
    # --- ROW 1: NO REID MODELS ---
    "Best MOTA (No ReID)":     {"folder": f"{MODEL_BASE_NAME}_NoReID_best_MOTA_TEST", "color": COLOR_MOTA,  "row": 1},
    "Best IDF1 (No ReID)":     {"folder": f"{MODEL_BASE_NAME}_NoReID_best_IDF1_TEST", "color": COLOR_IDF1,  "row": 1},
    "Final Epoch (No ReID)":   {"folder": f"{MODEL_BASE_NAME}_NoReID_TEST",           "color": COLOR_FINAL, "row": 1},
}
 
# ==============================================================================
# --- PUBLICATION-QUALITY PLOT STYLING (Q1 Journal Standards) ---
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",       # Academic Times style serif font
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "grid.alpha": 0.25,
    "text.usetex": False          # Clean portability across standard server terminals
})

# Thresholds 0.05 to 0.95 (Step size 0.05)
thresholds = np.arange(0.05, 1.00, 0.05) 
csv_columns_det = [f"DetA___{int(round(t*100))}" for t in thresholds]
csv_columns_ass = [f"AssA___{int(round(t*100))}" for t in thresholds]

# Create a 2x2 grid matrix sharing both X and Y dimensions globally
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)

# Loop and distribute models to their designated grid panels
for label, config in TARGET_SETUPS.items():
    folder_name = config["folder"]
    csv_path = os.path.join(RESULTS_ROOT, folder_name, TRACKER_SUBPATH, folder_name, "pedestrian_detailed.csv")
    
    if not os.path.exists(csv_path):
        print(f"[!] Warning: Path missing, skipping: {csv_path}")
        continue
        
    df = pd.read_csv(csv_path)
    df_combined = df[df["seq"].str.lower() == "combined"]
    if df_combined.empty: df_combined = df.iloc[[-1]]
        
    det_curves = df_combined[csv_columns_det].values.flatten()
    ass_curves = df_combined[csv_columns_ass].values.flatten()
    
    r = config["row"]
    # Column 0 is always Detection Accuracy (DetA), Column 1 is Association Accuracy (AssA)
    axes[r, 0].plot(thresholds, det_curves, color=config["color"], linewidth=2.0, marker="o",markersize=4, label=label)
    axes[r, 1].plot(thresholds, ass_curves, color=config["color"], linewidth=2.0, marker="x",markersize=5, linestyle="--", label=label)

# ------------------------------------------------------------------------------
# --- AXES & LABEL POLISHING ---
# ------------------------------------------------------------------------------
# Top Row Labels (With ReID)
axes[0, 0].set_title("Detection Accuracy ($DetA$ Curve) - With ReID", pad=12)
axes[0, 1].set_title("Association Accuracy ($AssA$ Curve) - With ReID", pad=12)
axes[0, 0].legend(loc="lower left", frameon=True)

# Bottom Row Labels (No ReID)
axes[1, 0].set_title("Detection Accuracy ($DetA$ Curve) - No ReID", pad=12)
axes[1, 1].set_title("Association Accuracy ($AssA$ Curve) - No ReID", pad=12)
axes[1, 0].legend(loc="lower left", frameon=True)

# Universal structural details across all 4 subplots
for row in range(2):
    for col in range(2):
        axes[row, col].set_xlim(0.05, 0.95)
        axes[row, col].set_ylim(0.0, 1.0)
        axes[row, col].set_xticks(np.arange(0.1, 1.0, 0.2))
        axes[row, col].grid(True, linestyle=":")
        
        # Apply labels selectively to the outer border plots to avoid interior crowding
        if row == 1:
            axes[row, col].set_xlabel("Localization Threshold (Alpha)")
        if col == 0:
            axes[row, col].set_ylabel("Accuracy Score")

# ------------------------------------------------------------------------------
# --- SAVE COMPOSITE PLOT ---
# ------------------------------------------------------------------------------
plt.tight_layout()

pdf_out = os.path.join(OUTPUT_GRAPH_DIR, "hota_curves_2x2_grid.pdf")
eps_out = os.path.join(OUTPUT_GRAPH_DIR, "hota_curves_2x2_grid.eps")
png_out = os.path.join(OUTPUT_GRAPH_DIR, "hota_curves_2x2_grid.png")

fig.savefig(pdf_out, bbox_inches="tight")
fig.savefig(eps_out, bbox_inches="tight")
fig.savefig(png_out, bbox_inches="tight", dpi=300)
plt.close()

print(f"\n[+] Success! 2x2 Matrix layout exported to:")
print(f"    -> {pdf_out}")
print(f"    -> {eps_out}")
