import os
import os.path as osp
import csv
from .mot17_sequence import MOT17Sequence
from ..coco import make_coco_transforms
from ..transforms import Compose

class EgohumansFullSequence(MOT17Sequence): # class name for full dataset
    data_folder = 'Egohumans_full_MOT' # <--- path to my custom MOT type dataset

    def __init__(self, root_dir: str = 'data', seq_name: str = None,
                 dets: str = '', vis_threshold: float = 0.0, img_transform = None) -> None:
        
        # <--- NEW: Manually initializing instead of calling super().__init__ 
        # to bypass the original train/test assertion error.
        
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        # <--- MODIFIED: Uses self.data_folder ('Egohumans_full_MOT') instead of 'MOT17'
        self._data_dir = osp.join(root_dir, self.data_folder)

        self._train_folders = os.listdir(osp.join(self._data_dir, 'train')) if osp.exists(osp.join(self._data_dir, 'train')) else []
        self._test_folders = os.listdir(osp.join(self._data_dir, 'test')) if osp.exists(osp.join(self._data_dir, 'test')) else []
        
        # <--- NEW: Added support for 'val' directory discovery
        val_path = osp.join(self._data_dir, 'val')
        self._val_folders = os.listdir(val_path) if osp.exists(val_path) else []

        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))

        self.data = []
        self.no_gt = True
        
        if seq_name is not None:
            full_seq_name = seq_name
            if self._dets is not None:
                full_seq_name = f"{seq_name}-{dets}"

            # <--- MODIFIED: Assertion now includes 'full_seq_name in self._val_folders'
            assert (full_seq_name in self._train_folders or 
                    full_seq_name in self._test_folders or 
                    full_seq_name in self._val_folders), \
                f'Image set does not exist in train/test/val: {full_seq_name}'

            self.data = self._sequence()
            self.no_gt = not osp.exists(self.get_gt_file_path())

    def _sequence(self):
        return super()._sequence()

    def get_seq_path(self) -> str:
        full_seq_name = self._seq_name
        if self._dets:
            full_seq_name = f"{self._seq_name}-{self._dets}"

        if full_seq_name in self._train_folders:
            return osp.join(self._data_dir, 'train', full_seq_name)
        
        # <--- NEW: Routing logic for the 'val' path
        elif full_seq_name in self._val_folders: 
            return osp.join(self._data_dir, 'val', full_seq_name)
        
        else:
            return osp.join(self._data_dir, 'test', full_seq_name)
"""
old script that ran successfully on baseline evaluation without traiing on these splits  "TRAIN and TEST"
"""
#import os
#import os.path as osp
#import csv
#from .mot17_sequence import MOT17Sequence

#class EgohumansSequence(MOT17Sequence):
#    """
#    Specialized loader for Egohumans_full that supports the 'val' split 
#    and flexible integer parsing for floating-point strings.
#    """
#    data_folder = 'Egohumans_full_MOT'
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
        
        # Add support for the 'val' directory which isn't in standard MOT17
#        val_path = osp.join(self._data_dir, 'val')
#        self._val_folders = os.listdir(val_path) if osp.exists(val_path) else [] # Claude
        

#    """
#    def _sequence(self):
#        # We override this to handle the float-to-int conversion in row[0]
#        # which you needed for some of your Egohumans/Egotracks annotations.

#        # self.seq_length = int(self.config['Sequence']['seqLength']) #  not needed.  
#        # The parent property already reads seqLength correctly from self.config, 
#        #  so the assignment is also completely redundant — it's doing exactly what the property already does.
#        dets = {i: [] for i in range(1, self.seq_length + 1)}
#        det_file = self.get_det_file_path()

#        if osp.exists(det_file):
#            with open(det_file, "r") as inf:
#                reader = csv.reader(inf, delimiter=',')
#                for row in reader:
#                    x1 = float(row[2]) - 1
#                    y1 = float(row[3]) - 1
#                    x2 = x1 + float(row[4]) - 1
#                    y2 = y1 + float(row[5]) - 1
#                    score = float(row[6])
#                    bbox = [x1, y1, x2, y2, score]
#                    # The safety fix you discovered earlier:
#                    dets[int(float(row[0]))].append(bbox)
#        
#        # Return total using the logic from the parent class
#        return super()._sequence()
#    """
#    def _sequence(self):
#    # seq_length is already provided by the parent @property
#    # reading from self.config['Sequence']['seqLength']
#    # No need to set it here.
#        return super()._sequence()


#    def get_seq_path(self) -> str:
#        full_seq_name = self._seq_name
#        if self._dets:
#            full_seq_name = f"{self._seq_name}-{self._dets}"

#        # Check train and test via parent, then check val
#        if full_seq_name in self._train_folders:
#            return osp.join(self._data_dir, 'train', full_seq_name)
#        elif hasattr(self, '_val_folders') and full_seq_name in self._val_folders:
#            return osp.join(self._data_dir, 'val', full_seq_name)
#        else:
#            return osp.join(self._data_dir, 'test', full_seq_name)


