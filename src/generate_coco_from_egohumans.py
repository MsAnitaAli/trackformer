import os
import json
import configparser
import csv
import shutil
import argparse
from pathlib import Path

def generate_coco_from_egohuman(data_root, split_name):
    """
    Converts MOT-style EgoHumans into Flattened COCO format for TrackFormer.
    """
    root_split_path = os.path.join(data_root, split_name)
    coco_dir = os.path.join(data_root, f"{split_name}_coco")
    
    # Setup Directories
    if not os.path.exists(coco_dir):
        os.makedirs(coco_dir)

    annotations_dir = os.path.join(data_root, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')

    coco_output = {
        "type": "instances",
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        "images": [],
        "annotations": [],
        "sequences": []
    }

    seqs = sorted([d for d in os.listdir(root_split_path) 
                   if os.path.isdir(os.path.join(root_split_path, d))])
    coco_output['sequences'] = seqs
    
    img_id = 0
    ann_id = 0
    img_name_to_id = {}

    print(f"Processing {split_name} split: {len(seqs)} sequences found.")

    for seq in seqs:
        seq_path = os.path.join(root_split_path, seq)
        config = configparser.ConfigParser()
        config.read(os.path.join(seq_path, 'seqinfo.ini'))
        
        img_width = int(config['Sequence']['imWidth'])
        img_height = int(config['Sequence']['imHeight'])
        seq_length = int(config['Sequence']['seqLength'])

        img_dir = os.path.join(seq_path, 'img1')
        images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        first_frame_image_id = img_id

        for i, img_file in enumerate(images):
            flat_img_name = f"{seq}_{img_file}"
            
            coco_output['images'].append({
                "file_name": flat_img_name,
                "height": img_height,
                "width": img_width,
                "id": img_id,
                "frame_id": i,
                "seq_length": seq_length,
                "first_frame_image_id": first_frame_image_id
            })
            
            img_name_to_id[flat_img_name] = img_id
            
            # Chained symlink logic
            src_path = os.path.abspath(os.path.join(img_dir, img_file))
            dst_path = os.path.join(coco_dir, flat_img_name)
            
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)
            
            img_id += 1

        # Process Ground Truth
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        if os.path.isfile(gt_path):
            with open(gt_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # MOT Format: [frame, id, x, y, w, h, conf, class, vis]
                    if int(row[7]) == 1 and int(row[6]) == 1:
                        frame_num = int(row[0])
                        current_img_name = f"{seq}_{frame_num:06d}.jpg"
                        target_img_id = img_name_to_id.get(current_img_name)
                        
                        if target_img_id is not None:
                            bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                            coco_output['annotations'].append({
                                "id": ann_id,
                                "bbox": [int(c) for c in bbox],
                                "image_id": target_img_id,
                                "segmentation": [],
                                "ignore": 0 if float(row[8]) > 0.25 else 1,
                                "visibility": float(row[8]),
                                "area": float(bbox[2] * bbox[3]),
                                "iscrowd": 0,
                                "seq": seq,
                                "category_id": 1,
                                "track_id": int(row[1])
                            })
                            ann_id += 1
        print(f"  [DONE] {seq}")

    with open(annotation_file, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"\nFinalized {split_name}: {img_id} images, {ann_id} annotations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate COCO from EgoHumans MOT.')
    parser.add_argument('--data_root', type=str, default='data/Egohumans_MOT')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    generate_coco_from_egohuman(args.data_root, args.split)
