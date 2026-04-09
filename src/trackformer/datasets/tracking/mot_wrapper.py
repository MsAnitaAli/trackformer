# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT wrapper which combines sequences to a dataset.
"""
from torch.utils.data import Dataset

from .mot17_sequence import MOT17Sequence
from .mot20_sequence import MOT20Sequence
from .mots20_sequence import MOTS20Sequence
from .egotracks_sequence import EgotracksSequence # added to handle egotracks


class MOT17Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, dets: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT17Sequence dataset
        """
        train_sequences = [
            'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
            'MOT17-10', 'MOT17-11', 'MOT17-13']
        test_sequences = [
            'MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
            'MOT17-08', 'MOT17-12', 'MOT17-14']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOT17-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT17-{split}"]
        else:
            raise NotImplementedError("MOT17 split not available.")

        self._data = []
        for seq in sequences:
            if dets == 'ALL':
                self._data.append(MOT17Sequence(seq_name=seq, dets='DPM', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='FRCNN', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='SDP', **kwargs))
            else:
                self._data.append(MOT17Sequence(seq_name=seq, dets=dets, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


class MOT20Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT20Sequence dataset
        """
        train_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05',]
        test_sequences = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08',]

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOT20-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT20-{split}"]
        else:
            raise NotImplementedError("MOT20 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(MOT20Sequence(seq_name=seq, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


class MOTS20Wrapper(MOT17Wrapper):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOTS20Sequence dataset
        """
        train_sequences = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
        test_sequences = ['MOTS20-01', 'MOTS20-06', 'MOTS20-07', 'MOTS20-12']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOTS20-{split}" in train_sequences + test_sequences:
            sequences = [f"MOTS20-{split}"]
        else:
            raise NotImplementedError("MOTS20 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(MOTS20Sequence(seq_name=seq, **kwargs))
            
"""
# This is added for egotracks during training phase
class EgotracksWrapper(Dataset):
    #A Wrapper for the EgotracksSequence class to return multiple sequences.

    def __init__(self, split: str, **kwargs) -> None:
        #Initializes the Egotracks dataset split.

        #Keyword arguments:
        #split -- the split of the dataset to use (TRAIN/VAL/TEST)
        #kwargs -- arguments passed to EgotracksSequence
        #
        # All video folders names are mentioned here
        train_sequences = [
            '1bc52b56-1e39-46df-be22-272480fd6022',
            '606919f7-3d65-4b2b-8351-b6ad4af97723'
        ]

        if split == "TRAIN" or split == "VAL":
            sequences = train_sequences
        elif split == "ALL":
            sequences = train_sequences
        else:
            # You can add specific TEST folders later if needed
            raise NotImplementedError(f"Egotracks split {split} not available.")

        self._data = []
        for seq in sequences:
            # We use dets=None because Egotracks doesn't use the MOT17 public det folders
            self._data.append(EgotracksSequence(seq_name=seq, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
"""

# This part is added  during evaluation
import os

class EgotracksWrapper(Dataset):
    """A generic Wrapper for Egotracks that scales automatically."""

    def __init__(self, split: str, seq: str = None, **kwargs) -> None:
        """
        Args:
            split: 'TRAIN', 'VAL', or 'TEST'
            seq: Optional specific ClipUID (provided by factory.py)
            kwargs: Arguments for EgotracksSequence
        """
        # Determine the source folder based on the split
        # We assume 'VAL' also comes from the 'train' folder for now
        subset = 'test' if split == 'TEST' else 'train'
        subset_path = os.path.join('data/egotracks', subset)
	# if train,tet and val folders are here then instaed of above two lines just use this  one
	# subset = split.lower()
        # Logic to determine which sequences to load
        if seq is not None:
            # If factory.py requested a specific clip, use only that one
            sequences = [seq]
        else:
            # Otherwise, scan the folder and load EVERYTHING in that split
            if os.path.exists(subset_path):
                sequences = [f for f in os.listdir(subset_path) 
                             if os.path.isdir(os.path.join(subset_path, f))]
                sequences.sort()
            else:
                sequences = []

        # 3. Initialize the sequences
        self._data = []
        for s in sequences:
            self._data.append(EgotracksSequence(seq_name=s, dets=None, **kwargs))

        if not self._data:
            print(f"Warning: No Egotracks sequences found for {split} in {subset_path}")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]

