# dataset settings
dataset_type = 'LaRSDataset'
data_root = 'data/lars/' # TODO: change this to your own path
ignore_idx=255
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=ignore_idx),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
crop_size = (768, 768)
ignore_idx = 255

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 640), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

dataset_type = 'LaRSDataset'
data_root = 'data/lars/'

# LaRS augmented training set
train_dataset = dict(
    type=dataset_type,
    data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
    pipeline=train_pipeline),

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
    pipeline=test_pipeline)

# data = dict(
#     samples_per_gpu=1, # NOTE: config made for 8 GPUs
#     workers_per_gpu=1, # NOTE: config made for 8 GPUs
#     train=[
#         dataset_lars_orig,
#         dataset_lars_aug
#     ],
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
train_dataloader = dict(batch_size=8,
                        num_workers=2,
                        persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        dataset=dict(
                            type=dataset_type,
                            data_root=data_root,
                            data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                            pipeline=train_pipeline))
val_dataloader = dict(batch_size=8,
                      num_workers=2,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(
                          type=dataset_type,
                          data_root=data_root,
                          data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                          pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
