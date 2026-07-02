!/bin/bash
#set -e. if this is used then it stops if one command fails. also remove || echo...

#ver 1. Epoch 50 — with reid
#python src/track_summary.py with reid \
#    dataset_name=Egohumans_full-TEST \
#    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_epoch_50.pth \
#    output_dir=results/temporal_memory_eval_reid_TEST \
#    write_images=False || echo "Command 1 failed, continuing..."
    
#ver 2. Epoch 50 — without reid
python src/track_summary.py with \
    dataset_name=Egohumans_full-TEST \
    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_epoch_50.pth \
    output_dir=results/temporal_memory_eval_Noreid_TEST \
    write_images=False || echo "Command 2 failed, continuing..."
    
#ver 3. Best IDF1 — with reid
python src/track_summary.py with reid \
    dataset_name=Egohumans_full-TEST \
    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_best_IDF1.pth \
    output_dir=results/temporal_memory_eval_reid_bestIDF1_TEST \
    write_images=False || echo "Command 3 failed, continuing..."
    
#ver 4. Best IDF1 — without reid
python src/track_summary.py with \
    dataset_name=Egohumans_full-TEST \
    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_best_IDF1.pth \
    output_dir=results/temporal_memory_eval_Noreid_bestIDF1_TEST \
    write_images=False || echo "Command 4 failed, continuing..."

#ver 5. Best MOTA — with reid
python src/track_summary.py with reid \
    dataset_name=Egohumans_full-TEST \
    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_best_MOTA.pth \
    output_dir=results/temporal_memory_eval_reid_bestMOTA_TEST \
    write_images=False || echo "Command 5 failed, continuing..."
    
#ver 6. Best MOTA — without reid
python src/track_summary.py with \
    dataset_name=Egohumans_full-TEST \
    obj_detect_checkpoint_file=models/egohumans_full/checkpoint_best_MOTA.pth \
    output_dir=results/temporal_memory_eval_Noreid_bestMOTA_TEST \
    write_images=False || echo "Command 6 failed, continuing..."
   
  echo "All 6 evaluations complete."
