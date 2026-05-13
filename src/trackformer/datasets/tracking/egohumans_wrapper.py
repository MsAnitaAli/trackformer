# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
EgoHumans wrapper which combines sequences to a dataset.
"""
from torch.utils.data import Dataset
from .egohumans_sequence import EgohumansSequence

class EgohumansWrapper(Dataset):
    """A Wrapper for the EgohumansSequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initializes all subset of the dataset."""
        
        train_sequences = [
            'fencing_004_fencing_aria01_rgb', 'fencing_004_fencing_aria02_rgb',
            'fencing_004_fencing_aria03_rgb', 'legoassemble_002_legoassemble_aria01_rgb',
            'legoassemble_002_legoassemble_aria02_rgb', 'legoassemble_002_legoassemble_aria03_rgb',
            'tagging_004_tagging_aria01_rgb', 'tagging_004_tagging_aria02_rgb',
            'tagging_004_tagging_aria03_rgb', 'tagging_004_tagging_aria04_rgb'
        ]

        val_sequences = [
            'fencing_013_fencing_aria01_rgb', 'fencing_013_fencing_aria03_rgb',
            'legoassemble_005_legoassemble_aria01_rgb', 
            'legoassemble_005_legoassemble_aria03_rgb', 'tagging_008_tagging_aria01_rgb',
            'tagging_008_tagging_aria03_rgb', 'tagging_008_tagging_aria04_rgb'
        ]

        test_sequences = [
            'fencing_014_fencing_aria03_rgb',
            'legoassemble_006_legoassemble_aria01_rgb', 'legoassemble_006_legoassemble_aria02_rgb',
            'legoassemble_006_legoassemble_aria03_rgb', 'tagging_013_tagging_aria01_rgb',
            'tagging_013_tagging_aria02_rgb', 'tagging_013_tagging_aria03_rgb',
            'tagging_013_tagging_aria04_rgb'
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
            raise NotImplementedError(f"EgoHumans split or sequence {split} not available.")

        self._data = []
        for seq in sequences:
            # We pass dets=None as EgoHumans baseline uses private detections/GT
            self._data.append(EgohumansSequence(seq_name=seq, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
