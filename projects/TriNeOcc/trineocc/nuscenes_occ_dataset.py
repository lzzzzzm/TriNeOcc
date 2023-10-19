# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

from mmengine.dataset import BaseDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class NuScenesOccDataset(BaseDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        test_mode (bool): Store `True` when building test or val dataset.
    """
    METAINFO = {
        'classes':
        ('noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation', 'free'),
        'ignore_index':17,
        'label_mapping':
        dict([(1, 0), (5, 0), (7, 0), (8, 0), (10, 0), (11, 0), (13, 0),
              (19, 0), (20, 0), (0, 0), (29, 0), (31, 0), (9, 1), (14, 2),
              (15, 3), (16, 3), (17, 4), (18, 5), (21, 6), (2, 7), (3, 7),
              (4, 7), (6, 7), (12, 8), (22, 9), (23, 10), (24, 11), (25, 12),
              (26, 13), (27, 14), (28, 15), (30, 16)]),
        'palette': [
            # [0,   0,   0, 255],  # 0 undefined
            [255, 158, 0, 255],  # 1 car  orange
            [0, 0, 230, 255],  # 2 pedestrian  Blue
            [47, 79, 79, 255],  # 3 sign  Darkslategrey
            [220, 20, 60, 255],  # 4 CYCLIST  Crimson
            [255, 69, 0, 255],  # 5 traiffic_light  Orangered
            [255, 140, 0, 255],  # 6 pole  Darkorange
            [233, 150, 70, 255],  # 7 construction_cone  Darksalmon
            [255, 61, 99, 255],  # 8 bycycle  Red
            [112, 128, 144, 255],  # 9 motorcycle  Slategrey
            [222, 184, 135, 255],  # 10 building Burlywood
            [0, 175, 0, 255],  # 11 vegetation  Green
            [165, 42, 42, 255],  # 12 trunk  nuTonomy green
            [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
            [75, 0, 75, 255],  # 14 walkable, sidewalk
            [255, 0, 0, 255],  # 15 unobsrvd
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs) -> None:
        metainfo = dict(label2cat={
            i: cat_name
            for i, cat_name in enumerate(self.METAINFO['classes'])
        })
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        data_list = []
        info['lidar_points']['lidar_path'] = \
            osp.join(
                self.data_prefix.get('pts', ''),
                info['lidar_points']['lidar_path'])

        for cam_id, img_info in info['images'].items():
            if 'img_path' in img_info:
                if cam_id in self.data_prefix:
                    cam_prefix = self.data_prefix[cam_id]
                else:
                    cam_prefix = self.data_prefix.get('img', '')
                img_info['img_path'] = osp.join(cam_prefix,
                                                img_info['img_path'])

        if 'pts_semantic_mask_path' in info:
            info['pts_semantic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_semantic_mask', ''),
                         info['pts_semantic_mask_path'])

        if 'occ_semantics_path' in info:
            info['occ_semantics_path'] = info['occ_semantics_path']


        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['seg_label_mapping'] = self.metainfo['label_mapping']

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode:
            info['eval_ann_info'] = dict()

        data_list.append(info)
        return data_list
