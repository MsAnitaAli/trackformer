# this version gave HOTA  results on terminal but was unable to wtite in file
"""
Standalone HOTA evaluation script for EgoHumans using TrackEval.

Place this file at:
    trackformer/src/evaluate_hota.py

Run from the trackformer root directory:

    # Evaluate test split (default)
    python src/evaluate_hota.py

    # Evaluate a specific split
    python src/evaluate_hota.py --split test
    python src/evaluate_hota.py --split val
    python src/evaluate_hota.py --split train

    # Evaluate with a custom tracker name (for comparing multiple models)
    python src/evaluate_hota.py --split test --tracker my_finetuned_model

What this script does:
  1. Verifies trackeval and numpy are correctly installed
  2. Creates a TrackEval-compatible folder structure using symlinks
     (no data duplication) under results/trackeval_workspace/
  3. Generates the required seqmaps file listing sequences for the split
  4. Runs TrackEval and reports HOTA, AssA, DetA, CLEAR, Identity metrics
     per sequence and overall
  5. Saves full results under results/trackeval_workspace/output/

Prerequisites (all already installed in trackformer_gpu env):
    trackeval==1.3.0  (install with: pip install trackeval --no-deps)
    numpy>=1.22.0,<1.24.0
    scipy, matplotlib, lap, lapsolver, pandas
"""
import argparse
import os
import sys
import shutil
import configparser
from datetime import datetime

# ── Repo root (script lives at src/evaluate_hota.py) ─────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Tracker results: where track.py saves <seq_name>.txt prediction files ────
TRACKER_RESULTS_DIR = os.path.join(
    REPO_ROOT, "results", "egohumans_baseline_visualization"
)

# ── MOT dataset root (contains train/ val/ test/ subfolders) ─────────────────
EGOHUMANS_MOT_ROOT = os.path.join(
    REPO_ROOT, "data", "Egohumans_MOT"
)

# ── Workspace where TrackEval symlink structure is built ─────────────────────
TRACKEVAL_WORKSPACE = os.path.join(
    REPO_ROOT, "results", "trackeval_workspace"
)


def check_environment() -> None:
    try:
        import numpy as np
        from packaging.version import Version
        v = Version(np.__version__)
        if not (Version("1.22.0") <= v < Version("1.24.0")):
            print(f"[WARN] numpy {np.__version__} is outside range >=1.22.0,<1.24.0.")
        else:
            print(f"[OK] numpy {np.__version__}")
    except ImportError:
        print("[ERROR] numpy not found.")
        sys.exit(1)

    try:
        import trackeval
        print("[OK] trackeval found.")
    except ImportError:
        print("[ERROR] trackeval not installed.")
        sys.exit(1)


def get_sequences(split: str) -> list:
    gt_split_dir = os.path.join(EGOHUMANS_MOT_ROOT, split)
    if not os.path.isdir(gt_split_dir):
        print(f"[ERROR] GT split folder not found: {gt_split_dir}")
        sys.exit(1)

    sequences = []
    for seq_name in sorted(os.listdir(gt_split_dir)):
        gt_path = os.path.join(gt_split_dir, seq_name, "gt", "gt.txt")
        tracker_path = os.path.join(TRACKER_RESULTS_DIR, f"{seq_name}.txt")

        if not os.path.exists(gt_path):
            continue
        if not os.path.exists(tracker_path):
            print(f"  [SKIP] {seq_name} — tracker output not found")
            continue
        sequences.append(seq_name)
    return sequences


def build_trackeval_structure(sequences: list, split: str, benchmark: str, tracker_name: str) -> tuple:
    gt_benchmark_dir = os.path.join(TRACKEVAL_WORKSPACE, "gt", "mot_challenge", benchmark)
    seqmaps_dir = os.path.join(gt_benchmark_dir, "seqmaps")
    os.makedirs(seqmaps_dir, exist_ok=True)
    
    seqmap_path = os.path.join(seqmaps_dir, f"{benchmark}.txt")
    with open(seqmap_path, "w") as f:
        f.write("name\n")
        for seq in sequences:
            f.write(f"{seq}\n")

    gt_split_dir = os.path.join(EGOHUMANS_MOT_ROOT, split)
    for seq in sequences:
        seq_root_dir = os.path.join(gt_benchmark_dir, seq)
        seq_gt_dir = os.path.join(seq_root_dir, "gt")
        os.makedirs(seq_gt_dir, exist_ok=True)

        src_gt = os.path.abspath(os.path.join(gt_split_dir, seq, "gt", "gt.txt"))
        dst_gt = os.path.join(seq_gt_dir, "gt.txt")
        if not os.path.exists(dst_gt):
            os.symlink(src_gt, dst_gt)

        src_ini = os.path.abspath(os.path.join(gt_split_dir, seq, "seqinfo.ini"))
        dst_ini = os.path.join(seq_root_dir, "seqinfo.ini")
        if not os.path.exists(dst_ini):
            os.symlink(src_ini, dst_ini)

    tracker_data_dir = os.path.join(TRACKEVAL_WORKSPACE, "trackers", "mot_challenge", benchmark, tracker_name, "data")
    os.makedirs(tracker_data_dir, exist_ok=True)

    for seq in sequences:
        src_txt = os.path.abspath(os.path.join(TRACKER_RESULTS_DIR, f"{seq}.txt"))
        dst_txt = os.path.join(tracker_data_dir, f"{seq}.txt") 
        if not os.path.exists(dst_txt):
            os.symlink(src_txt, dst_txt)

    return os.path.join(TRACKEVAL_WORKSPACE, "gt", "mot_challenge"), \
           os.path.join(TRACKEVAL_WORKSPACE, "trackers", "mot_challenge")


def run_trackeval(gt_dir: str, tracker_dir: str, benchmark: str, tracker_name: str, sequences: list) -> dict:
    import trackeval
    output_folder = os.path.join(TRACKEVAL_WORKSPACE, "output")
    
    # ─── FIX: Map sequence name directly to the INTEGER length ───
    seq_info_dict = {}
    for seq in sequences:
        ini_path = os.path.join(gt_dir, benchmark, seq, 'seqinfo.ini')
        
        config = configparser.ConfigParser()
        config.read(ini_path)
        
        if 'Sequence' in config and 'seqLength' in config['Sequence']:
            # TrackEval expects the value to be the integer frame count 
            # if it's not a path string.
            seq_info_dict[seq] = int(config['Sequence']['seqLength'])
        else:
            print(f"[WARN] seqLength not found in {ini_path}, falling back to path.")
            seq_info_dict[seq] = ini_path

    eval_config = {
        **trackeval.Evaluator.get_default_eval_config(), 
        "PRINT_RESULTS": True, 
        "OUTPUT_FOLDER": output_folder
    }

    dataset_config = {
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
        "GT_FOLDER": gt_dir, 
        "TRACKERS_FOLDER": os.path.join(tracker_dir, benchmark), 
        "BENCHMARK": benchmark, 
        "SPLIT_TO_EVAL": benchmark,
        "TRACKERS_TO_EVAL": [tracker_name], 
        "CLASSES_TO_EVAL": ["pedestrian"],
        "DO_PREPROC": False, 
        "SKIP_SPLIT_FOL": True,
        "TRACKER_SUB_FOLDER": "data",
        "TRACKER_EXT": ".txt",
        "GT_LOC_FORMAT": "{gt_folder}/" + benchmark + "/{seq}/gt/gt.txt",
        "SEQ_INFO": seq_info_dict, # Now contains: {'seq_name': 331, ...}
        "SEQMAP_FILE": os.path.join(gt_dir, benchmark, "seqmaps", f"{benchmark}.txt")
    }
    
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    metrics_list = [getattr(trackeval.metrics, name)(metrics_config) for name in metrics_config["METRICS"]]
    
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    res, _ = evaluator.evaluate([dataset], metrics_list)
    return res    
"""    
def write_summary(res, benchmark, tracker_name, sequences, split):
    import numpy as np
    summary_dir = os.path.join(REPO_ROOT, "results", "mot_summaries")
    os.makedirs(summary_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    summary_path = os.path.join(summary_dir, f"hota_results_{benchmark}_{timestamp}.txt")

    dataset_key = list(res.keys())[0]
    tracker_res = res[dataset_key][tracker_name]
    CLS = "pedestrian"

    lines = ["="*90, f" EgoHumans HOTA Summary | Benchmark: {benchmark} | Tracker: {tracker_name}", "="*90]
    header = f"{'Sequence':<52} {'HOTA':>6} {'AssA':>6} {'DetA':>6} {'MOTA':>7} {'IDF1':>6}"
    lines.append(header)
    lines.append("-" * 90)

    all_metrics = []
    for seq in sequences:
        if seq in tracker_res and seq != 'COMBINED_SEQ':
            sr = tracker_res[seq]
            h = np.mean(sr["HOTA"][CLS]["HOTA"]) * 100
            a = np.mean(sr["HOTA"][CLS]["AssA"]) * 100
            d = np.mean(sr["HOTA"][CLS]["DetA"]) * 100
            m = sr["CLEAR"][CLS]["MOTA"] * 100
            i = sr["Identity"][CLS]["IDF1"] * 100
            all_metrics.append([h, a, d, m, i])
            lines.append(f"{seq:<52} {h:>6.1f} {a:>6.1f} {d:>6.1f} {m:>7.1f} {i:>6.1f}")

    if all_metrics:
        avg = np.mean(all_metrics, axis=0)
        lines.append("-" * 90)
        lines.append(f"{'OVERALL':<52} {avg[0]:>6.1f} {avg[1]:>6.1f} {avg[2]:>6.1f} {avg[3]:>7.1f} {avg[4]:>6.1f}")
    
    output_text = "\n".join(lines)
    print(f"\n{output_text}\n\n[INFO] Summary saved: {summary_path}")
    with open(summary_path, "w") as f:
        f.write(output_text)

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--tracker", type=str, default="trackformer_mot20_baseline")
    args = parser.parse_args()

    if os.path.exists(TRACKEVAL_WORKSPACE):
        shutil.rmtree(TRACKEVAL_WORKSPACE)

    check_environment()
    seqs = get_sequences(args.split)
    if not seqs:
        print(f"No sequences found for split {args.split}. Check {TRACKER_RESULTS_DIR}")
        sys.exit(1)
    
    benchmark = f"Egohumans-{args.split.upper()}"
    gt_d, tr_d = build_trackeval_structure(seqs, args.split, benchmark, args.tracker)
    
    print(f"\n[INFO] Running evaluation for {benchmark}...")
    results = run_trackeval(gt_d, tr_d, benchmark, args.tracker, seqs)
    #write_summary(results, benchmark, args.tracker, seqs, args.split)

