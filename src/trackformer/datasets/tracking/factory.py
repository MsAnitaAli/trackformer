# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Factory of tracking datasets.
"""
from typing import Union

from torch.utils.data import ConcatDataset

from .demo_sequence import DemoSequence
from .mot_wrapper import MOT17Wrapper, MOT20Wrapper, MOTS20Wrapper
from .egohumans_wrapper import EgohumansWrapper
from .egohumans_full_wrapper import EgohumansFullWrapper # added  to handle Egohumans_full dataset

DATASETS = {}

# ── MOT17 ─────────────────────────────────────────────────────────────────────
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12', '13', '14']:
    for dets in ['DPM', 'FRCNN', 'SDP', 'ALL']:
        name = f'MOT17-{split}'
        if dets:
            name = f"{name}-{dets}"
        DATASETS[name] = (
            lambda kwargs, split=split, dets=dets: MOT17Wrapper(split, dets, **kwargs))

# ── MOT20 ─────────────────────────────────────────────────────────────────────
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05', '06', '07', '08']:
    name = f'MOT20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOT20Wrapper(split, **kwargs))

# ── MOTS20 ────────────────────────────────────────────────────────────────────
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '05', '06', '07', '09', '11', '12']:
    name = f'MOTS20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOTS20Wrapper(split, **kwargs))

# ── EgoHumans (subset) ────────────────────────────────────────────────────────
for split in ['TRAIN', 'VAL', 'TEST', 'ALL']:
    name = f'Egohumans-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: EgohumansWrapper(split, **kwargs))

# Individual sequences for subset
egohuman_seqs = [
    'fencing_004_fencing_aria01_rgb', 'fencing_004_fencing_aria02_rgb',
    'fencing_004_fencing_aria03_rgb', 'legoassemble_002_legoassemble_aria01_rgb',
    'legoassemble_002_legoassemble_aria02_rgb', 'legoassemble_002_legoassemble_aria03_rgb',
    'tagging_004_tagging_aria01_rgb', 'tagging_004_tagging_aria02_rgb',
    'tagging_004_tagging_aria03_rgb', 'tagging_004_tagging_aria04_rgb',
    'fencing_013_fencing_aria01_rgb', 'fencing_013_fencing_aria03_rgb',
    'legoassemble_005_legoassemble_aria01_rgb',
    'legoassemble_005_legoassemble_aria03_rgb', 'tagging_008_tagging_aria01_rgb',
    'tagging_008_tagging_aria03_rgb', 'tagging_008_tagging_aria04_rgb',
    'fencing_014_fencing_aria03_rgb',
    'legoassemble_006_legoassemble_aria01_rgb', 'legoassemble_006_legoassemble_aria02_rgb',
    'legoassemble_006_legoassemble_aria03_rgb', 'tagging_013_tagging_aria01_rgb',
    'tagging_013_tagging_aria02_rgb', 'tagging_013_tagging_aria03_rgb',
    'tagging_013_tagging_aria04_rgb',
]
for seq in egohuman_seqs:
    DATASETS[seq] = (
        lambda kwargs, seq=seq: EgohumansWrapper(seq, **kwargs))

# ── EgoHumans Full ────────────────────────────────────────────────────────────
for split in ['TRAIN', 'VAL', 'TEST', 'ALL']:
    name = f'Egohumans_full-{split}' ## Modified for egohumans_full
    DATASETS[name] = (
        lambda kwargs, split=split: EgohumansFullWrapper(split, **kwargs)) # Modified for egohumans_full

# Individual sequences for full dataset
egohuman_full_seqs = [
	    'tagging_001_tagging_aria01_rgb', 'tagging_001_tagging_aria02_rgb',
            'tagging_001_tagging_aria03_rgb', 'tagging_001_tagging_aria04_rgb',
            'tagging_002_tagging_aria01_rgb', 'tagging_002_tagging_aria02_rgb',
            'tagging_002_tagging_aria03_rgb', 'tagging_002_tagging_aria04_rgb',
            'tagging_003_tagging_aria01_rgb', 'tagging_003_tagging_aria02_rgb',
            'tagging_003_tagging_aria03_rgb', 'tagging_003_tagging_aria04_rgb',
            'tagging_004_tagging_aria01_rgb', 'tagging_004_tagging_aria02_rgb',
            'tagging_004_tagging_aria03_rgb', 'tagging_004_tagging_aria04_rgb',
            'tagging_005_tagging_aria01_rgb', 'tagging_005_tagging_aria02_rgb',
            'tagging_005_tagging_aria03_rgb', 'tagging_005_tagging_aria04_rgb',
            'tagging_006_tagging_aria01_rgb', 'tagging_006_tagging_aria02_rgb',
            'tagging_006_tagging_aria03_rgb', 'tagging_006_tagging_aria04_rgb',
            'tagging_007_tagging_aria01_rgb', 'tagging_007_tagging_aria02_rgb',
            'tagging_007_tagging_aria03_rgb', 'tagging_007_tagging_aria04_rgb',
            'tagging_008_tagging_aria01_rgb', 
            'tagging_008_tagging_aria03_rgb', 'tagging_008_tagging_aria04_rgb',
            'tagging_009_tagging_aria01_rgb', 
            'tagging_009_tagging_aria03_rgb', 'tagging_009_tagging_aria04_rgb',
            'tagging_010_tagging_aria01_rgb', 
            'tagging_010_tagging_aria03_rgb', 'tagging_010_tagging_aria04_rgb',
            'legoassemble_001_legoassemble_aria02_rgb',
            'legoassemble_003_legoassemble_aria01_rgb', 'legoassemble_003_legoassemble_aria02_rgb',
            'legoassemble_003_legoassemble_aria03_rgb',
            'legoassemble_004_legoassemble_aria01_rgb', 'legoassemble_004_legoassemble_aria02_rgb',
            'legoassemble_005_legoassemble_aria01_rgb', 
            'legoassemble_005_legoassemble_aria03_rgb',
            'fencing_001_fencing_aria01_rgb', 'fencing_001_fencing_aria02_rgb',
            'fencing_001_fencing_aria03_rgb',
            'fencing_002_fencing_aria01_rgb', 'fencing_002_fencing_aria02_rgb',
            'fencing_002_fencing_aria03_rgb',
            'fencing_003_fencing_aria01_rgb', 'fencing_003_fencing_aria02_rgb',
            'fencing_003_fencing_aria03_rgb',
            'fencing_004_fencing_aria01_rgb', 'fencing_004_fencing_aria02_rgb',
            'fencing_004_fencing_aria03_rgb',
            'fencing_005_fencing_aria01_rgb', 'fencing_005_fencing_aria02_rgb',
            'fencing_005_fencing_aria03_rgb',
            'fencing_006_fencing_aria01_rgb', 'fencing_006_fencing_aria02_rgb',
            'fencing_006_fencing_aria03_rgb',
            'fencing_008_fencing_aria01_rgb', 'fencing_008_fencing_aria02_rgb',
            'fencing_008_fencing_aria03_rgb',
            'fencing_009_fencing_aria01_rgb', 'fencing_009_fencing_aria02_rgb',
            'fencing_009_fencing_aria03_rgb',
            'fencing_011_fencing_aria03_rgb',
            'fencing_014_fencing_aria03_rgb',
            'tagging_011_tagging_aria01_rgb', 
            'tagging_011_tagging_aria03_rgb', 'tagging_011_tagging_aria04_rgb',
            'tagging_012_tagging_aria01_rgb', 'tagging_012_tagging_aria02_rgb',
            'tagging_012_tagging_aria03_rgb', 'tagging_012_tagging_aria04_rgb',
            'legoassemble_002_legoassemble_aria01_rgb', 'legoassemble_002_legoassemble_aria02_rgb',
            'legoassemble_002_legoassemble_aria03_rgb',
            'fencing_012_fencing_aria01_rgb',
            'fencing_012_fencing_aria03_rgb',            
            'fencing_013_fencing_aria01_rgb',
            'fencing_013_fencing_aria03_rgb' ,
            'tagging_013_tagging_aria01_rgb', 'tagging_013_tagging_aria02_rgb',
            'tagging_013_tagging_aria03_rgb', 'tagging_013_tagging_aria04_rgb',
            'tagging_014_tagging_aria01_rgb', 'tagging_014_tagging_aria02_rgb',
            'tagging_014_tagging_aria03_rgb', 'tagging_014_tagging_aria04_rgb', 
            'legoassemble_006_legoassemble_aria01_rgb',   'legoassemble_006_legoassemble_aria02_rgb',
            'legoassemble_006_legoassemble_aria03_rgb',
            'fencing_010_fencing_aria01_rgb', 'fencing_010_fencing_aria02_rgb',
            'fencing_010_fencing_aria03_rgb'
            ]

for seq in egohuman_full_seqs:
    DATASETS[seq] = (
        lambda kwargs, seq=seq: EgohumansFullWrapper(seq, **kwargs))

# ── Demo ──────────────────────────────────────────────────────────────────────
DATASETS['DEMO'] = (lambda kwargs: [DemoSequence(**kwargs), ])


class TrackDatasetFactory:
    """A central class to manage the individual dataset loaders."""

    def __init__(self, datasets: Union[str, list], **kwargs) -> None:
        if isinstance(datasets, str):
            datasets = [datasets]

        self._data = None
        for dataset in datasets:
            assert dataset in DATASETS, f"[!] Dataset not found: {dataset}"

            if self._data is None:
                self._data = DATASETS[dataset](kwargs)
            else:
                self._data = ConcatDataset(
                    [self._data, DATASETS[dataset](kwargs)]
                )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
