_base_ = ['./segformer_mit-b0_8x1_1024x1024_160k_lars.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa
crop_size = (768, 768)
ignore_idx = 255

data_preprocessor = dict(size=crop_size)
test_cfg = None
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512)))