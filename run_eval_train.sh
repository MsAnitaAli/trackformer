!/bin/bash
#set -e. if this is used then it stops if one command fails. also remove || echo...

# ________Already run, so commented out ____________  
#1. Epoch 50 — with reid
#python src/track_summary.py with \
#    reid \
#     dataset_name=Egohumans_full-TRAIN \
#    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_epoch_50.pth \
#    output_dir=results/egohumans_full_evaluation_TRAIN_reid \
#    write_images=False || echo "Command 1 failed, continuing..."


# 2. Epoch 50 — without reid
#python src/track_summary.py with \
#    dataset_name=Egohumans_full-TRAIN \
#    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_epoch_50.pth \
#    output_dir=results/egohumans_full_evaluation_TRAIN_noreid \
#    write_images=False || echo "Command 2 failed, continuing..."
    

# 3. Best IDF1 — with reid
python src/track_summary.py with \
    reid \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_best_IDF1.pth \
    output_dir=results/egohumans_full_evaluation_TRAIN_reid_best_IDF1 \
    write_images=False || echo "Command 3 failed, continuing..."

# 4. Best IDF1 — without reid
python src/track_summary.py with \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_best_IDF1.pth \
    output_dir=results/egohumans_full_evaluation_TRAIN_noreid_best_IDF1 \
    write_images=False || echo "Command 4 failed, continuing..."

# 5. Best MOTA — with reid
python src/track_summary.py with \
    reid \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_best_MOTA.pth \
    output_dir=results/egohumans_full_evaluation_TRAIN_reid_best_MOTA \
    write_images=False || echo "Command 5 failed, continuing..."

# 6. Best MOTA — without reid
python src/track_summary.py with \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_best_MOTA.pth \
    output_dir=results/egohumans_full_evaluation_TRAIN_noreid_best_MOTA \
    write_images=False || echo "Command 6 failed, continuing..."
    
  
echo "All 6 evaluations complete."
