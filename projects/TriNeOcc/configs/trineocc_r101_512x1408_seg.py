_base_ = ['./trineocc_r101_512x1408_occ.py']

dataset_type = 'NuScenesOccDataset'
data_root = 'data/nuscenes/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-mini',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

ida_aug_conf = {
        "resize_lim": (0.8, 1.0),
        "final_dim": (512, 1408),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        # "rand_flip": False,
        "rand_flip": False,
    }
backend_args = None

val_pipeline = [
    dict(
        type='BEVOccLoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_occ_3d=True,
        with_attr_label=False,
        seg_3d_dtype='np.uint8'),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='SegLabelMapping'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'pts_semantic_mask'],
        meta_keys=['lidar2img', 'ego2lidar', 'lidar2ego'])
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = val_dataloader

test_pipeline = val_pipeline
val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='CustomHook', pts_seg_enable=False)
]

model = dict(
    type='TriNeOcc',
    predict_task='seg'
)