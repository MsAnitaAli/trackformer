# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
EgoHumans Full dataset wrapper which combines sequences to a dataset.
"""
from torch.utils.data import Dataset
from .egohumans_full_sequence import EgohumansFullSequence # modified for full sequence


class EgohumansFullWrapper(Dataset): # modified for full sequence
    """A Wrapper for the EgohumansFullSequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initializes all sequences for the requested split."""

        train_sequences = [
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
            'fencing_014_fencing_aria03_rgb'
        ]

        val_sequences = [
            'tagging_011_tagging_aria01_rgb', 
            'tagging_011_tagging_aria03_rgb', 'tagging_011_tagging_aria04_rgb',
            'tagging_012_tagging_aria01_rgb', 'tagging_012_tagging_aria02_rgb',
            'tagging_012_tagging_aria03_rgb', 'tagging_012_tagging_aria04_rgb',
            'legoassemble_002_legoassemble_aria01_rgb', 'legoassemble_002_legoassemble_aria02_rgb',
            'legoassemble_002_legoassemble_aria03_rgb',
            'fencing_012_fencing_aria01_rgb',
            'fencing_012_fencing_aria03_rgb',            
            'fencing_013_fencing_aria01_rgb',
            'fencing_013_fencing_aria03_rgb'            
        ]

        test_sequences = [
            'tagging_013_tagging_aria01_rgb', 'tagging_013_tagging_aria02_rgb',
            'tagging_013_tagging_aria03_rgb', 'tagging_013_tagging_aria04_rgb',
            'tagging_014_tagging_aria01_rgb', 'tagging_014_tagging_aria02_rgb',
            'tagging_014_tagging_aria03_rgb', 'tagging_014_tagging_aria04_rgb', 
            'legoassemble_006_legoassemble_aria01_rgb',   'legoassemble_006_legoassemble_aria02_rgb',
            'legoassemble_006_legoassemble_aria03_rgb',
            'fencing_010_fencing_aria01_rgb', 'fencing_010_fencing_aria02_rgb',
            'fencing_010_fencing_aria03_rgb',
             
        ]

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "VAL":
            sequences = val_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + val_sequences + test_sequences
        elif split in train_sequences + val_sequences + test_sequences:
            sequences = [split]
        else:
            raise NotImplementedError(
                f"EgoHumans_full split or sequence '{split}' not available."  # error handling attached here
            )

        self._data = []
        for seq in sequences:
            self._data.append(
                EgohumansFullSequence(seq_name=seq, dets=None, **kwargs) # modified for full sequence
            )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]

