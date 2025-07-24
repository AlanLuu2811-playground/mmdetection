_base_ = './tood_r50_fpn_1x_coco.py'
model = dict(bbox_head=dict(anchor_type='anchor_based'))
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            # 'language_model': dict(lr_mult=0),
        }))
