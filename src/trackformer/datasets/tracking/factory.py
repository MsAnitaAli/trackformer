# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Factory of tracking datasets.
"""
from typing import Union

from torch.utils.data import ConcatDataset

from .demo_sequence import DemoSequence
from .mot_wrapper import MOT17Wrapper, MOT20Wrapper, MOTS20Wrapper, EgotracksWrapper

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

DATASETS['DEMO'] = (lambda kwargs: [DemoSequence(**kwargs), ])

"""
#  this part is added to access egotracks data
# this part was used during training phase
for split in ['TRAIN', 'VAL', 'TEST']:
    name = f'mot-egotracks-{split}'  # Added 'mot-' prefix 
    # Use EgotracksWrapper directly instead of MOT17Wrapper
    DATASETS[name] = (
        lambda kwargs, split=split: EgotracksWrapper(split, **kwargs)
    )
# egootracks info ends here
"""
# this part was added during evaluation phase to handle all egotracks clips
import  os
EGOTRACKS_ROOT = 'data/egotracks'

for split in ['TRAIN', 'VAL', 'TEST']:
    name = f'mot-egotracks-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: EgotracksWrapper(split, **kwargs)
    )

# We scan both train and test folders to find clip uids
for subset in ['train', 'test']:
    subset_path = os.path.join(EGOTRACKS_ROOT, subset)
    
    if os.path.exists(subset_path):
        # Find all subdirectories (these are your ClipUIDs)
        clip_ids = [f for f in os.listdir(subset_path) 
                     if os.path.isdir(os.path.join(subset_path, f))]
        
        for cid in clip_ids:
            # Map the ClipUID directly to a dataset name
            # We pass subset.upper() so the Wrapper knows if it's TRAIN or TEST
            DATASETS[cid] = (
                lambda kwargs, cid=cid, subset=subset: EgotracksWrapper(
                    subset.upper(), seq=cid, **kwargs)
            )

#ends here


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
