import trackeval
import os

# 1. Define where your results are stored
# This should be the parent folder containing your 6 tracker folders
RESULTS_PARENT = os.path.abspath("results/all_hota_ped_summaries") 

# 2. List the folder names of your 6 experiments
tracker_list = [
    "TrackEgoFormer_ReID",
    "TrackEgoFormer_NoReID",
    "TrackEgoFormer_ReID_best_IDF1",
    "TrackEgoFormer_NoReID_best_IDF1",
    "TrackEgoFormer_ReID_best_MOTA",
    "TrackEgoFormer_NoReID_best_MOTA"
]

# 3. Define where the combined plots should be saved
output_folder = os.path.join(RESULTS_PARENT, "comparison_plots")

# 4. Run the comparison plotter
# This calls the code you modified in plotting.py!
trackeval.plotting.plot_compare_trackers(
    tracker_folder=RESULTS_PARENT,
    tracker_list=tracker_list,
    cls="pedestrian",
    output_folder=output_folder
)

print(f"Comparison plots generated in: {output_folder}")
