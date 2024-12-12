import os.path as osp

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class LaRSDataset(BaseSegDataset):
    """LaRS dataset.

    LaRS dataset contains maritime scenes with obstacle, water and sky masks.
    """

    METAINFO = dict(classes = ('obstacle',
                               'water',
                               'sky'),
                    palette = [[247, 195, 37],
                               [41, 167, 224],
                               [90, 75, 164]])

    def __init__(self, data_root=None, split=None, **kwargs):
        super(LaRSDataset, self).__init__(
            data_root=data_root,
            img_suffix='.jpg',
            seg_map_suffix='.png',
            ignore_index=255,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(data_root)