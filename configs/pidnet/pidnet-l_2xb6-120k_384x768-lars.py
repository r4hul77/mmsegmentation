_base_ = './pidnet-s_2xb6-120k_384x768-lars.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-l_imagenet1k_20230306-67889109.pth'  # noqa

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=1, save_best=['mIoU'], rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))


model = dict(
    backbone=dict(
        channels=64,
        ppm_channels=112,
        num_stem_blocks=3,
        num_branch_blocks=4,
        init_cfg=dict(checkpoint=checkpoint_file)),
    decode_head=dict(in_channels=256, channels=256))
