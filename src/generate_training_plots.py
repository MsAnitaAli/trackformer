"""
This code reads training log file and generates these 2 graphs as pdf and esp. standard format for Q1 jounrnals
1. total_loss_epochs : The clean, smooth overview line tracking overall training progression epoch-by-epoch
2. component_losses_iteration: The detailed, multi-line iteration chart breaking down individual loss_ce, loss_bbox, and loss_giou tracking components
How to run: python src/generate_training_plots.py my_model_name
my_model_name can be replaced with my model name, if other than egohumans_appearance. as, it is set as deafult is in code
"""
import os
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd

# 1. PARAMETER AND INPUT ARGUMENT HANDLING
# Fallback default value if no arguments are explicitly supplied
model_name = "egohumans_appearance"# default model name or can be passed from command line

# If an argument is provided (e.g. python src/generate_training_plots.py multi_frame_deformable)
if len(sys.argv) > 1:
    model_name = sys.argv[1]

LOG_FILE_PATH = f"models/{model_name}/log.txt" # actual
#LOG_FILE_PATH = f"models/{model_name}/log_1_20epochs.txt" # used this to generate apperaance ver2 graphs
OUTPUT_DIR = f"results/{model_name}_graphs"

# Verification checkpoint
if not os.path.exists(LOG_FILE_PATH):
    print(f"\n[!] Error: Log target missing at: {LOG_FILE_PATH}")
    print("Usage template: python src/generate_training_plots.py [your_model_dir_name]")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. DEFINING REGEX TARGET PATTERNS
log_pattern = re.compile(
    r"Epoch:\s+\[(?P<epoch>\d+)\]\s+\[\s*(?P<iter>\d+)/\d+\].*?"
    r"loss:\s+\S+\s+\((?P<loss>\S+)\).*?"
    r"loss_ce:\s+\S+\s+\((?P<loss_ce>\S+)\).*?"
    r"loss_bbox:\s+\S+\s+\((?P<loss_bbox>\S+)\).*?"
    r"loss_giou:\s+\S+\s+\((?P<loss_giou>\S+)\)"
)

parsed_data = {}

# 3. STREAMING LOG LINES AND REMOVING DUPLICATES
print(f"Reading and parsing: {LOG_FILE_PATH}...")
with open(LOG_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "Epoch: [" in line and "loss:" in line:
            match = log_pattern.search(line)
            if match:
                epoch = int(match.group("epoch"))
                iteration = int(match.group("iter"))
                
                # step_key unique tuple manages the collision cleanup logic
                step_key = (epoch, iteration)
                
                # Overwrites old tracking iterations from power drop entries seamlessly
                parsed_data[step_key] = {
                    "Epoch": epoch,
                    "Iteration": iteration,
                    "Total_Loss": float(match.group("loss")),
                    "Classification_Loss": float(match.group("loss_ce")),
                    "BBox_Loss": float(match.group("loss_bbox")),
                    "GIoU_Loss": float(match.group("loss_giou")),
                }

if not parsed_data:
    print("[!] Error: No training lines could be extracted. Check the internal syntax of the file.")
    sys.exit(1)

# Build, sort and index dataframe object sequentially
df = pd.DataFrame(parsed_data.values()).sort_values(by=["Epoch", "Iteration"]).reset_index(drop=True)

# Generate a continuous step axis sequence across epochs
df["Global_Step"] = (df["Epoch"] - 1) * 33027 + df["Iteration"]

print(f"Extracted {len(df)} unique tracking steps. Rendering figures...")

# 4. DESIGN RULES AND BACKEND STYLING (COMPATIBLE WITH MATPLOTLIB 3.5.3)
plt.style.use("seaborn-whitegrid")  # Explicitly matches my environment version style
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    "font.size": 11, 
    "axes.labelsize": 12, 
    "xtick.labelsize": 10, 
    "ytick.labelsize": 10
})

# --- GRAPH 1: CONVERGENCE PROGRESSION OVER TOTAL EPOCHS ---
df_epoch = df.groupby("Epoch")["Total_Loss"].last().reset_index()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df_epoch["Epoch"], df_epoch["Total_Loss"], color="#1f77b4", linewidth=2.0, marker="o", markersize=5)
ax.set_xlabel("Epochs")
ax.set_ylabel("Total Loss (Running Average)")
ax.set_xticks(df_epoch["Epoch"])

plt.tight_layout()
#fig.savefig(os.path.join(OUTPUT_DIR, "total_loss_epochs.pdf"), bbox_inches="tight")
#fig.savefig(os.path.join(OUTPUT_DIR, "total_loss_epochs.eps"), bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "total_loss_epochs(20epochs).pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "total_loss_epochs(20epochs).eps"), bbox_inches="tight")
plt.close()


# --- GRAPH 2: GRANULAR SUB-LOSS COMPONENT SPLIT ---
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(df["Global_Step"], df["Classification_Loss"], color="#d62728", linewidth=1.2, label="Classification Loss (loss_ce)")
ax.plot(df["Global_Step"], df["BBox_Loss"], color="#2ca02c", linewidth=1.2, label="Bounding Box Loss (loss_bbox)")
ax.plot(df["Global_Step"], df["GIoU_Loss"], color="#ff7f0e", linewidth=1.2, label="GIoU Loss (loss_giou)")

ax.set_xlabel("Global Iteration Steps")
ax.set_ylabel("Loss Component Values")
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()
#fig.savefig(os.path.join(OUTPUT_DIR, "component_losses_iteration.pdf"), bbox_inches="tight")
#fig.savefig(os.path.join(OUTPUT_DIR, "component_losses_iteration.eps"), bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "component_losses_iteration(20epochs).pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "component_losses_iteration(20epochs).eps"), bbox_inches="tight")
plt.close()


print(f"Success! Q1-Compliant Vector Assets dropped here: '{OUTPUT_DIR}/'")
