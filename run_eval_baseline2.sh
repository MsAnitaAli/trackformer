#!/bin/bash
# 1 is done through command prompt
#1. TEST split
#python src/track_summary.py with \
#    reid \
#    dataset_name=Egohumans_full-TEST \
#    obj_detect_checkpoint_file=models/mot20_crowdhuman_deformable_multi_frame/checkpoint_epoch_50.pth \
#    output_dir=results/egohumans_baseline2_visualization_reid_TEST \
#    write_images=False
# 2. VAL split
python src/track_summary.py with \
    reid \
    dataset_name=Egohumans_full-VAL \
    obj_detect_checkpoint_file=models/mot20_crowdhuman_deformable_multi_frame/checkpoint_epoch_50.pth \
    output_dir=results/egohumans_baseline2_visualization_reid_VAL \
    write_images=False || echo "Command 2 failed, continuing..."


# 3. TRAIN split
python src/track_summary.py with \
    reid \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/mot20_crowdhuman_deformable_multi_frame/checkpoint_epoch_50.pth \
    output_dir=results/egohumans_baseline2_visualization_reid_TRAIN \
    write_images=False || echo "Command 3 failed, continuing..."
# 4. ALL
python src/track_summary.py with \
    reid \
    dataset_name=Egohumans_full-ALL \
    obj_detect_checkpoint_file=models/mot20_crowdhuman_deformable_multi_frame/checkpoint_epoch_50.pth \
    output_dir=results/egohumans_baseline2_visualization_reid_ALL \
    write_images=False || echo "Command 4 failed, continuing..."

echo "All 4 evaluations complete."
