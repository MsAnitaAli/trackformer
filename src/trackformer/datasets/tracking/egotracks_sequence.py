# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
egotracks dataset.
"""

from .mot17_sequence import MOT17Sequence

class EgotracksSequence(MOT17Sequence):
    """Egotracks Dataset.

    This dataloader handles one sequence at a time by inheriting from MOT17.
    """
    data_folder = 'egotracks'
