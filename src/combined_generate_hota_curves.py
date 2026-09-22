"""
This code reads training log file and generates HOTA Curves graph as pdf and esp. standard format for Q1 journals
HOTA Curves: (DetA vs. AssA): Plot Detection Accuracy ($DetA$) and Association Accuracy ($AssA$) against localization thresholds. This specific graph requires the full HOTA breakdown to see whether your tracker's main bottleneck is object detection or frame-to-frame identity matching.
How to run: from main branch: run this:  --> python src/combined_generate_hota_curves.py
I am moving results of different branches at same place so that I can plotHOTA curve for baseline, fintuned model and appearance plan
"""

# ***********************************
# New code - use scale0 to 1  
#*************************************
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_GRAPH_DIR = "results/egohumans_graphs"
os.makedirs(MODEL_GRAPH_DIR, exist_ok=True)

# Dataset paths and distinct visual configurations
TARGET_SETUPS = {
    "Baseline": {
        "path": "results/TrackEgoFormer_Baseline2_ReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormer_Baseline2_ReID_TEST/pedestrian_detailed.csv",
        "color": "#d62728",     # Red
        "linestyle": ":",       # Dotted line
        "marker": "s"           # Square marker
    },
    "EgoTracker": {
        "path": "results/TrackEgoFormer_ReID/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormer_ReID/pedestrian_detailed.csv",
        "color": "#1f77b4",     # Blue
        "linestyle": "--",      # Dashed line
        "marker": "^"           # Triangle marker
    },
    "EgoAppTracker(Ours)": {
        "path": "results/EgoAppTracker_ReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/EgoAppTracker_ReID_TEST/pedestrian_detailed.csv",
        "color": "#2ca02c",     # Green
        "linestyle": "-",       # Solid line
        "marker": "o"           # Circle marker
    }
}

# Publication-quality styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "grid.alpha": 0.25,
    "text.usetex": False
})

# Thresholds (0.05 to 0.95) with robust IEEE 754 precision rounding
thresholds = np.arange(0.05, 1.00, 0.05) 
csv_columns_det = [f"DetA___{int(round(t*100))}" for t in thresholds]
csv_columns_ass = [f"AssA___{int(round(t*100))}" for t in thresholds]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

for label, cfg in TARGET_SETUPS.items():
    csv_path = cfg["path"]
    if not os.path.exists(csv_path):
        print(f"[!] Warning: Target CSV missing at path: {csv_path}. Skipping variant.")
        continue
        
    df = pd.read_csv(csv_path)
    df_combined = df[df["seq"].str.lower() == "combined"]
    if df_combined.empty:
        df_combined = df.iloc[[-1]]
        
    det_curves = df_combined[csv_columns_det].values.flatten()
    ass_curves = df_combined[csv_columns_ass].values.flatten()
    
    # Plot Detection Accuracy (DetA)
    ax1.plot(thresholds, det_curves, color=cfg["color"], linewidth=1.8, 
             linestyle=cfg["linestyle"], marker=cfg["marker"], markevery=2, 
             markersize=5, label=label, alpha=0.9)
    
    # Plot Association Accuracy (AssA)
    ax2.plot(thresholds, ass_curves, color=cfg["color"], linewidth=1.8, 
             linestyle=cfg["linestyle"], marker=cfg["marker"], markevery=2, 
             markersize=5, label=label, alpha=0.9)

# --- SUBPLOT 1: DETECTION ACCURACY ---
ax1.set_xlabel(r"Localization Threshold ($\alpha$)")
ax1.set_ylabel("Accuracy Score")
ax1.set_title(r"Detection Accuracy ($DetA$ Curve)", pad=10)
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)
ax1.set_xticks(np.arange(0.0, 1.1, 0.1))
ax1.set_yticks(np.arange(0.0, 1.1, 0.1))
ax1.grid(True, linestyle=":")
ax1.legend(loc="lower left", frameon=True)

# --- SUBPLOT 2: ASSOCIATION ACCURACY ---
ax2.set_xlabel(r"Localization Threshold ($\alpha$)")
ax2.set_title(r"Association Accuracy ($AssA$ Curve)", pad=10)
ax2.set_xlim(0.0, 1.0)
ax2.set_xticks(np.arange(0.0, 1.1, 0.1))
ax2.grid(True, linestyle=":")
ax2.legend(loc="lower left", frameon=True)

plt.tight_layout()

# Save publication files
fig.savefig(os.path.join(MODEL_GRAPH_DIR, "hota_curves_breakdown.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(MODEL_GRAPH_DIR, "hota_curves_breakdown.eps"), bbox_inches="tight")
fig.savefig(os.path.join(MODEL_GRAPH_DIR, "hota_curves_breakdown.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"[+] Full scale diagnostic plot saved to {MODEL_GRAPH_DIR}")

# ***********************************
# New code - graph seems to be trimmed 
#*************************************
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_GRAPH_DIR = "results/egohumans_graphs"
os.makedirs(MODEL_GRAPH_DIR, exist_ok=True)

# Dataset paths
TARGET_SETUPS = {
    "Baseline": {
        "path": "results/TrackEgoFormer_Baseline2_ReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormer_Baseline2_ReID_TEST/pedestrian_detailed.csv",
        "color": "#d62728",     # Red
        "linestyle": ":",       # Dotted
        "marker": "s"           # Square
    },
    "EgoTracker": {
        "path": "results/TrackEgoFormer_ReID/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormer_ReID/pedestrian_detailed.csv",
        "color": "#1f77b4",     # Blue
        "linestyle": "--",      # Dashed (lets underlying solid line show through)
        "marker": "^"           # Triangle
    },
    "EgoAppTracker (Proposed)": {
        "path": "results/EgoAppTracker_ReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/EgoAppTracker_ReID_TEST/pedestrian_detailed.csv",
        "color": "#2ca02c",     # Green
        "linestyle": "-",       # Solid
        "marker": "o"           # Circle
    }
}

marker="x",markersize=5, linestyle="--"
# Publication-quality styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "grid.alpha": 0.25,
    "text.usetex": False
})

# Thresholds with IEEE 754 precision fix
thresholds = np.arange(0.05, 1.00, 0.05) 
csv_columns_det = [f"DetA___{int(round(t*100))}" for t in thresholds]
csv_columns_ass = [f"AssA___{int(round(t*100))}" for t in thresholds]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

all_det_vals, all_ass_vals = [], []

for label, cfg in TARGET_SETUPS.items():
    csv_path = cfg["path"]
    if not os.path.exists(csv_path):
        print(f"[!] Warning: Target CSV missing at path: {csv_path}. Skipping variant.")
        continue
        
    df = pd.read_csv(csv_path)
    df_combined = df[df["seq"].str.lower() == "combined"]
    if df_combined.empty:
        df_combined = df.iloc[[-1]]
        
    det_curves = df_combined[csv_columns_det].values.flatten()
    ass_curves = df_combined[csv_columns_ass].values.flatten()
    
    all_det_vals.extend(det_curves)
    all_ass_vals.extend(ass_curves)
    
    # Plot Detection Accuracy (DetA)
    # Using markevery=2 prevents marker crowding over 19 thresholds
    ax1.plot(thresholds, det_curves, color=cfg["color"], linewidth=1.8, 
             linestyle=cfg["linestyle"], marker=cfg["marker"], markevery=2, 
             markersize=5, label=label, alpha=0.9)
    
    # Plot Association Accuracy (AssA)
    ax2.plot(thresholds, ass_curves, color=cfg["color"], linewidth=1.8, 
             linestyle=cfg["linestyle"], marker=cfg["marker"], markevery=2, 
             markersize=5, label=label, alpha=0.9)

# --- SUBPLOT 1: DETECTION ACCURACY ---
ax1.set_xlabel(r"Localization Threshold ($\alpha$)")
ax1.set_ylabel("Accuracy Score")
ax1.set_title(r"Detection Accuracy ($DetA$ Curve)", pad=10)
ax1.set_xlim(0.05, 0.95)
ax1.grid(True, linestyle=":")
ax1.legend(loc="lower left", frameon=True)

# --- SUBPLOT 2: ASSOCIATION ACCURACY ---
ax2.set_xlabel(r"Localization Threshold ($\alpha$)")
ax2.set_title(r"Association Accuracy ($AssA$ Curve)", pad=10)
ax2.set_xlim(0.05, 0.95)
ax2.grid(True, linestyle=":")
ax2.legend(loc="lower left", frameon=True)

# --- AXIS SCALING ADJUSTMENT ---
# Tighten Y-axis limits dynamically based on actual data range to remove white space
if all_det_vals and all_ass_vals:
    min_val = min(min(all_det_vals), min(all_ass_vals))
    max_val = max(max(all_det_vals), max(all_ass_vals))
    # Add a 5% margin
    ax1.set_ylim(max(0.0, min_val - 0.05), min(1.0, max_val + 0.05))

plt.tight_layout()

fig.savefig(os.path.join(MODEL_GRAPH_DIR, "hota_curves_breakdown_legible.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(MODEL_GRAPH_DIR, "hota_curves_breakdown_legible.eps"), bbox_inches="tight")
fig.savefig(os.path.join(MODEL_GRAPH_DIR, "hota_curves_breakdown_legible.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"[+] Legible diagnostic comparison plot saved to {MODEL_GRAPH_DIR}")
"""
#****************************
# OLD Code
#***************************
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION (Change these paths or experiment variants dynamically) ---
MODEL_GRAPH_DIR = "results/egohumans_graphs"
os.makedirs(MODEL_GRAPH_DIR, exist_ok=True)

# Dictionary pointing to your respective experiment evaluation outputs
# Key: Plot Display Label | Value: Target CSV location path

TARGET_SETUPS = {
    "Baseline": "results/TrackEgoFormer_Baseline2_ReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormer_Baseline2_ReID_TEST/pedestrian_detailed.csv",
    "EgoTracker": "results/TrackEgoFormer_ReID/trackers/mot_challenge/Egohumans_full_MOT-TEST/TrackEgoFormer_ReID/pedestrian_detailed.csv",
    "EgoAppTracker": "results/EgoAppTracker_ReID_TEST/trackers/mot_challenge/Egohumans_full_MOT-TEST/EgoAppTracker_ReID_TEST/pedestrian_detailed.csv"
}


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
"""
