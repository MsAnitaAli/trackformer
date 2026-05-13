# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Factory of tracking datasets.
"""
from typing import Union

from torch.utils.data import ConcatDataset

from .demo_sequence import DemoSequence
from .mot_wrapper import MOT17Wrapper, MOT20Wrapper, MOTS20Wrapper
from .egohumans_wrapper import EgohumansWrapper  # added  to handle Egohumans dataset

DATASETS = {}

# Fill all available datasets, change here to modify / add new datasets.
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12', '13', '14']:
    for dets in ['DPM', 'FRCNN', 'SDP', 'ALL']:
        name = f'MOT17-{split}'
        if dets:
            name = f"{name}-{dets}"
        DATASETS[name] = (
            lambda kwargs, split=split, dets=dets: MOT17Wrapper(split, dets, **kwargs))


for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08']:
    name = f'MOT20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOT20Wrapper(split, **kwargs))


for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '05', '06', '07', '09', '11', '12']:
    name = f'MOTS20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOTS20Wrapper(split, **kwargs))

"""
Added to esatblaish baseline evaluation on Egohumans 
"""
# Register EgoHumans Dataset
for split in ['TRAIN', 'VAL', 'TEST', 'ALL']:
    name = f'Egohumans-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: EgohumansWrapper(split, **kwargs))

# Also allow direct access to individual sequences
# I have added all  sequences. It works for both  Egohumans and Egohumans_full
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
    'tagging_013_tagging_aria04_rgb'    
]

for seq in egohuman_seqs:
    DATASETS[seq] = (lambda kwargs, seq=seq: EgohumansWrapper(seq, **kwargs))

"""
Logic for Egohumans ends here
"""
DATASETS['DEMO'] = (lambda kwargs: [DemoSequence(**kwargs), ])

class TrackDatasetFactory:
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, datasets: Union[str, list], **kwargs) -> None:
        """Initialize the corresponding dataloader.

        Keyword arguments:
        datasets --  the name of the dataset or list of dataset names
        kwargs -- arguments used to call the datasets
        """
        if isinstance(datasets, str):
            datasets = [datasets]

        self._data = None
        for dataset in datasets:
            assert dataset in DATASETS, f"[!] Dataset not found: {dataset}"

            if self._data is None:
                self._data = DATASETS[dataset](kwargs)
            else:
                self._data = ConcatDataset([self._data, DATASETS[dataset](kwargs)])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]

