"""
Generates Q1-compliant training plots from TrackFormer log files.

Graphs produced:
  1. total_loss_epochs.pdf/eps/png     — Train vs Val total loss per epoch
  2. component_losses_iteration.pdf/eps/png — Sub-loss components over iterations (train only)

Usage:
  python src/generate_training_plots.py [model_name]

Default model_name: egohumans_appearance
"""
import os
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd

# ── 1. PARAMETERS ─────────────────────────────────────────────────────────────
model_name = "egohumans_appearance"
if len(sys.argv) > 1:
    model_name = sys.argv[1]

LOG_FILE_PATH = f"models/{model_name}/log.txt"
OUTPUT_DIR    = f"results/{model_name}_graphs"

if not os.path.exists(LOG_FILE_PATH):
    print(f"\n[!] Error: Log file missing: {LOG_FILE_PATH}")
    print("Usage: python src/generate_training_plots.py [model_dir_name]")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 2. REGEX PATTERNS ─────────────────────────────────────────────────────────
train_pattern = re.compile(
    r"Epoch:\s+\[(?P<epoch>\d+)\]\s+\[\s*(?P<iter>\d+)/\d+\].*?"
    r"loss:\s+\S+\s+\((?P<loss>\S+)\).*?"
    r"loss_ce:\s+\S+\s+\((?P<loss_ce>\S+)\).*?"
    r"loss_bbox:\s+\S+\s+\((?P<loss_bbox>\S+)\).*?"
    r"loss_giou:\s+\S+\s+\((?P<loss_giou>\S+)\)"
)

# Val lines: "Epoch: [Test:]  [iter/total] ... loss: X (avg) ..."
# We extract the running average (parenthesised value) at the last iteration
# to get the epoch-level val loss.
val_pattern = re.compile(
    r"Epoch:\s+\[Test:\]\s+\[\s*(?P<iter>\d+)/(?P<total>\d+)\].*?"
    r"loss:\s+\S+\s+\((?P<loss>\S+)\).*?"
    r"loss_ce:\s+\S+\s+\((?P<loss_ce>\S+)\).*?"
    r"loss_bbox:\s+\S+\s+\((?P<loss_bbox>\S+)\).*?"
    r"loss_giou:\s+\S+\s+\((?P<loss_giou>\S+)\)"
)

# ── 3. PARSE LOG ──────────────────────────────────────────────────────────────
print(f"Parsing: {LOG_FILE_PATH} ...")

train_data = {}     # (epoch, iter) → dict
val_last   = {}     # epoch_counter → last Test: line seen

# We track val epochs by order of appearance in the log
val_epoch_counter = 0
current_val_epoch = None

with open(LOG_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:

        # ── Train lines ──────────────────────────────────────────────────────
        if "Epoch: [" in line and "loss:" in line and "Test:" not in line:
            m = train_pattern.search(line)
            if m:
                epoch = int(m.group("epoch"))
                itr   = int(m.group("iter"))
                train_data[(epoch, itr)] = {
                    "Epoch":                 epoch,
                    "Iteration":             itr,
                    "Total_Loss":            float(m.group("loss")),
                    "Classification_Loss":   float(m.group("loss_ce")),
                    "BBox_Loss":             float(m.group("loss_bbox")),
                    "GIoU_Loss":             float(m.group("loss_giou")),
                }

        # ── Val (Test:) lines ────────────────────────────────────────────────
        elif "Epoch: [Test:]" in line and "loss:" in line:
            m = val_pattern.search(line)
            if m:
                itr   = int(m.group("iter"))
                total = int(m.group("total"))

                # First test line of a new val run → increment epoch counter
                if itr == 0:
                    val_epoch_counter += 1
                    current_val_epoch = val_epoch_counter

                # Keep updating so we end up with the final-iteration average
                if current_val_epoch is not None:
                    val_last[current_val_epoch] = {
                        "Val_Loss":    float(m.group("loss")),
                        "Val_Loss_ce": float(m.group("loss_ce")),
                        "Val_Loss_bbox": float(m.group("loss_bbox")),
                        "Val_Loss_giou": float(m.group("loss_giou")),
                    }

if not train_data:
    print("[!] No training lines found. Check log file format.")
    sys.exit(1)

# Build train dataframe
df_train = (pd.DataFrame(train_data.values())
              .sort_values(["Epoch", "Iteration"])
              .reset_index(drop=True))
df_train["Global_Step"] = (df_train["Epoch"] - 1) * df_train["Iteration"].max() + df_train["Iteration"]

# Per-epoch train loss (last running-average value in each epoch)
df_train_epoch = df_train.groupby("Epoch")["Total_Loss"].last().reset_index()

# Val dataframe — map val_epoch_counter back to real training epochs
# Val evaluations happen at epoch 1 and every val_interval epochs.
# Infer the mapping from how many val epochs we found vs total train epochs.
total_train_epochs = df_train_epoch["Epoch"].max()
val_epochs_found   = sorted(val_last.keys())

# Try to match val epoch counter to train epoch numbers
# Strategy: if train epochs = [1,5,10,15] and we have 4 val runs → map directly
train_val_epochs = sorted(df_train_epoch["Epoch"].unique())
# val_interval = 5 is common; use the actual sequence of train epochs with val
# Simpler: just label val epochs 1..N and note in the graph
val_records = [{"Val_Epoch_Idx": k, "Val_Loss": v["Val_Loss"]}
               for k, v in sorted(val_last.items())]
df_val = pd.DataFrame(val_records)

print(f"Train steps parsed: {len(df_train)}")
print(f"Val evaluations found: {len(df_val)}")

# ── 4. STYLING ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-whitegrid")
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42
plt.rcParams.update({
    "font.size":       11,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ── 5. GRAPH 1 — TRAIN vs VAL LOSS PER EPOCH ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

# Train loss
ax.plot(df_train_epoch["Epoch"],
        df_train_epoch["Total_Loss"],
        color="#1f77b4", linewidth=2.0, marker="o", markersize=5,
        label="Training Loss")

# Val loss — align val epoch index to training epoch axis
# Map val index 1,2,3,... to actual training epochs where val was run
# Use val_interval spacing inferred from data
if len(df_val) > 0:
    # If we have val at epoch 1 then every val_interval, reconstruct epochs
    if total_train_epochs > 0 and len(val_epochs_found) > 0:
        # Assign real epoch numbers to val evaluations
        # epoch 1 always has val; subsequent ones at val_interval
        val_real_epochs = [e for e in sorted(train_val_epochs)
                           if e == 1 or (e % max(1, total_train_epochs // max(1, len(val_epochs_found) - 1))) == 0]
        # Fallback: just use evenly spaced
        if len(val_real_epochs) != len(val_epochs_found):
            import numpy as np
            val_real_epochs = list(np.linspace(1, total_train_epochs, len(val_epochs_found), dtype=int))

        df_val["Train_Epoch"] = val_real_epochs[:len(df_val)]

        ax.plot(df_val["Train_Epoch"],
                df_val["Val_Loss"],
                color="#d62728", linewidth=2.0, marker="s", markersize=5,
                linestyle="--", label="Validation Loss")

ax.set_xlabel("Epoch")
ax.set_ylabel("Total Loss (Running Average)")
ax.set_xticks(df_train_epoch["Epoch"])
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()
for ext in ["pdf", "eps", "png"]:
    fig.savefig(os.path.join(OUTPUT_DIR, f"train_test_loss_epochs.{ext}"),
                bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close()
print("Graph 1 saved: train_test_loss_epochs")
"""
# ── 6. GRAPH 2 — COMPONENT LOSSES OVER ITERATIONS (TRAIN ONLY) ───────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(df_train["Global_Step"], df_train["Classification_Loss"],
        color="#d62728", linewidth=1.2, label="Classification Loss (loss_ce)")
ax.plot(df_train["Global_Step"], df_train["BBox_Loss"],
        color="#2ca02c", linewidth=1.2, label="Bounding Box Loss (loss_bbox)")
ax.plot(df_train["Global_Step"], df_train["GIoU_Loss"],
        color="#ff7f0e", linewidth=1.2, label="GIoU Loss (loss_giou)")

ax.set_xlabel("Global Iteration Steps")
ax.set_ylabel("Loss Component Values")
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()
for ext in ["pdf", "eps", "png"]:
    fig.savefig(os.path.join(OUTPUT_DIR, f"component_losses_iteration.{ext}"),
                bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close()
print("Graph 2 saved: component_losses_iteration")
"""
print(f"\nDone. Outputs in: '{OUTPUT_DIR}/'")
