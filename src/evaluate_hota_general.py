# this version gave HOTA  results on terminal but was unable to write in file. we will copy terminal output
import argparse
import os
import sys
import shutil
import configparser
import trackeval

# Repo root (Assumes script is in src/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_environment():
    try:
        import numpy as np
        from packaging.version import Version
        v = Version(np.__version__)
        if not (Version("1.22.0") <= v < Version("1.26.0")):
            print(f"[WARN] numpy {np.__version__} is outside tested range.")
        print(f"[OK] Environment verified (TrackEval {trackeval.__version__ if hasattr(trackeval, '__version__') else 'found'}).")
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        sys.exit(1)

def get_sequences(data_root, split):
    gt_split_dir = os.path.join(data_root, split)
    if not os.path.isdir(gt_split_dir):
        print(f"[ERROR] GT split folder not found: {gt_split_dir}")
        sys.exit(1)
    # Return sorted list of subdirectories that contain a gt/gt.txt
    seqs = [d for d in sorted(os.listdir(gt_split_dir)) 
            if os.path.exists(os.path.join(gt_split_dir, d, "gt", "gt.txt"))]
    return seqs

def build_trackeval_structure(sequences, split, benchmark, tracker_name, data_root, tracker_results_dir):
    """Builds the temporary symlink structure TrackEval requires."""
    WORKSPACE = os.path.join(REPO_ROOT, "results", "trackeval_workspace")
    if os.path.exists(WORKSPACE):    
        print("going to delete trackeval_workspace")       
        shutil.rmtree(WORKSPACE)
        
    # Paths for GT and Seqmaps
    gt_benchmark_dir = os.path.join(WORKSPACE, "gt", "mot_challenge", benchmark)
    seqmaps_dir = os.path.join(gt_benchmark_dir, "seqmaps")
    os.makedirs(seqmaps_dir, exist_ok=True)
    
    # 1. Create the .txt seqmap file
    seqmap_path = os.path.join(seqmaps_dir, f"{benchmark}.txt")
    with open(seqmap_path, "w") as f:
        f.write("name\n")
        for seq in sequences:
            f.write(f"{seq}\n")

    # 2. Link Ground Truth and SeqInfo
    gt_split_dir = os.path.join(data_root, split)
    for seq in sequences:
        seq_root_dir = os.path.join(gt_benchmark_dir, seq)
        os.makedirs(os.path.join(seq_root_dir, "gt"), exist_ok=True)

        # Symlink gt.txt
        src_gt = os.path.abspath(os.path.join(gt_split_dir, seq, "gt", "gt.txt"))
        dst_gt = os.path.join(seq_root_dir, "gt", "gt.txt")
        if os.path.exists(src_gt): os.symlink(src_gt, dst_gt)

        # Symlink seqinfo.ini
        src_ini = os.path.abspath(os.path.join(gt_split_dir, seq, "seqinfo.ini"))
        dst_ini = os.path.join(seq_root_dir, "seqinfo.ini")
        if os.path.exists(src_ini): os.symlink(src_ini, dst_ini)

    # 3. Link Tracker Results
    tracker_data_dir = os.path.join(WORKSPACE, "trackers", "mot_challenge", benchmark, tracker_name, "data")
    os.makedirs(tracker_data_dir, exist_ok=True)

    for seq in sequences:
        # Note: some trackers save results as seq_name.txt, others as seq_name-None.txt
        # We try to find the match in the user-provided results_dir
        src_txt = os.path.abspath(os.path.join(tracker_results_dir, f"{seq}.txt"))
        
        # Fallback for filenames with suffixes (like the ones in your previous MOT output)
        if not os.path.exists(src_txt):
            src_txt = os.path.abspath(os.path.join(tracker_results_dir, f"{seq}-None.txt"))

        dst_txt = os.path.join(tracker_data_dir, f"{seq}.txt") 
        if os.path.exists(src_txt):
            os.symlink(src_txt, dst_txt)
        else:
            print(f"[SKIP] Tracker output missing for {seq} at {src_txt}")

    return WORKSPACE

def run_trackeval(workspace, benchmark, tracker_name, sequences):
    output_folder = os.path.join(workspace, "output")
    gt_dir = os.path.join(workspace, "gt", "mot_challenge")
    tracker_dir = os.path.join(workspace, "trackers", "mot_challenge")
    
    # Extract frame lengths for HOTA calculations
    seq_info_dict = {}
    for seq in sequences:
        ini_path = os.path.join(gt_dir, benchmark, seq, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(ini_path)
        if 'Sequence' in config and 'seqLength' in config['Sequence']:
            seq_info_dict[seq] = int(config['Sequence']['seqLength'])

    eval_config = {
        **trackeval.Evaluator.get_default_eval_config(), 
        "PRINT_RESULTS": True, 
        "OUTPUT_FOLDER": output_folder,
        "DISPLAY_LESS_PROGRESS": True
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
        "GT_LOC_FORMAT": "{gt_folder}/" + benchmark + "/{seq}/gt/gt.txt",
        "SEQ_INFO": seq_info_dict,
        "SEQMAP_FILE": os.path.join(gt_dir, benchmark, "seqmaps", f"{benchmark}.txt")
    }
    
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    metrics_list = [getattr(trackeval.metrics, name)(metrics_config) for name in metrics_config["METRICS"]]
    
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    evaluator.evaluate([dataset], metrics_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic HOTA Evaluator")
    parser.add_argument("--dataset", type=str, required=True, help="e.g., Egohumans or Egohumans_full")
    parser.add_argument("--split", type=str, default="test", help="train, val, or test")
    parser.add_argument("--results_dir", type=str, required=True, help="Folder containing tracker .txt files")
    parser.add_argument("--tracker_name", type=str, default="my_tracker", help="Label for results table")
    args = parser.parse_args()

    # Dynamic Data Pathing
    data_root = os.path.join(REPO_ROOT, "data", args.dataset) # just pass dataset folder name as a flag
    
    benchmark = f"{args.dataset}-{args.split.upper()}"
    
    check_environment()
    seqs = get_sequences(data_root, args.split)
    
    if not seqs:
        print(f"[ERROR] No sequences with valid GT found in {data_root}/{args.split}")
        sys.exit(1)

    workspace = build_trackeval_structure(seqs, args.split, benchmark, args.tracker_name, data_root, args.results_dir)
    
    print(f"\n[START] Evaluating {args.tracker_name} on {benchmark}...")
    run_trackeval(workspace, benchmark, args.tracker_name, seqs)
