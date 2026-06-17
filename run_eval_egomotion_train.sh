!/bin/bash
#set -e. if this is used then it stops if one command fails. also remove || echo...

#ver 1. best IDF1 with reid
python src/track_summary.py with reid \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_egomotion/checkpoint_best_IDF1.pth \
    output_dir=results/egomotion_eval_reid_bestIDF1_TRAIN \
    write_images=False || echo "Command 1 failed, continuing..."

#ver 2. best IDF1 — with no reid
python src/track_summary.py with \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_egomotion/checkpoint_best_IDF1.pth \
    output_dir=results/egomotion_eval_Noreid_bestIDF1_TRAIN \
    write_images=False || echo "Command 2 failed, continuing..."

#ver 3. Best MOTA — with reid
python src/track_summary.py with reid \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_egomotion/checkpoint_best_MOTA.pth \
    output_dir=results/egomotion_eval_reid_bestMOTA_TRAIN \
    write_images=False || echo "Command 3 failed, continuing..."
    
#ver 4. Best MOTA — with no reid
python src/track_summary.py with \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_egomotion/checkpoint_best_MOTA.pth \
    output_dir=results/egomotion_eval_Noreid_bestMOTA_TRAIN \
    write_images=False || echo "Command 4 failed, continuing..."

#ver 5. Epoch 10 — with reid
python src/track_summary.py with reid \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_egomotion/checkpoint_epoch_10.pth \
    output_dir=results/egomotion_eval_reid_TRAIN \
    write_images=False || echo "Command 5 failed, continuing..."

#ver 6. Epoch 10 — with no reid
python src/track_summary.py with \
    dataset_name=Egohumans_full-TRAIN \
    obj_detect_checkpoint_file=models/egohumans_egomotion/checkpoint_epoch_10.pth \
    output_dir=results/egomotion_eval_Noreid_TRAIN \
    write_images=False || echo "Command 6 failed, continuing..."
   
  echo "All 6 evaluations complete."
