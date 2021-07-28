# ### ===============================================================
# ### ===============================================================
# ### Modify the dataset loading settings

# dataset settings
dataset_type = 'ContrastDataset'
data_root = '/mnt/cadlabnas/datasets/' 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromNiiFile'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
    #dict(
        #type='Rotate',
        #level=1,
        #max_rotate_angle=7,
        #img_fill_val=0,
        #random_negative_prob=0.5,
        #prob=0.5
        #),
    dict(
        type='RandomCrop',
        #crop_type='absolute_range',
        crop_size=(512,512),
        #allow_negative_crop=False
    ),
]
test_pipeline = [
    dict(type='LoadImageFromNiiFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=1,        # BATCH_SIZE
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file='train.txt',
            data_prefix= data_root + 'RenalDonors/',
            pipeline=train_pipeline),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file='val.txt',
        data_prefix= data_root + 'RenalDonors/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='test.txt',
        data_prefix= data_root + 'RenalDonors/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='accuracy', metric_options=dict(topk=(1,)))

# Set up working dir to save files and logs.
work_dir = '/home/alec/Desktop/ImgClassification/working_dir'


### ===============================================================
### ===============================================================
### Modify the model settings

# model settings
model = dict(
    type='ImageClassifier',
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))


### ===============================================================
### ===============================================================
### Modify the schedule settings

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 4 since we only use one GPU.
# optimizer
optimizer_lr = 0.001 #0.01 / 4

# optimizer
optimizer = dict(type='SGD', lr=optimizer_lr, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=0.001,
    step=[5, 10])
runner = dict(type='EpochBasedRunner', max_epochs=25)


### ===============================================================
### ===============================================================
### Modify the default runtime settings

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50, #50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None


# run train iter 1 time (overall 1 time which includes: div num_images by batch_size, and mult by dataset_repeat_times)
# run validation iter 1 time 
# only setting workflow = [('train', 1)] will not backpropagate validation error/loss through the network  
workflow = [('train', 1), ('val', 1)]


### ===============================================================
### ===============================================================
### Miscellaneous settings

# Set seed thus the results are more reproducible
seed = 0
#set_random_seed(0, deterministic=False)
gpu_ids = range(1)


### ===============================================================
### ===============================================================
### testing/prediction/evaluation phase - Model settings 

# get the root path to the model checkpoints
ckp_root = work_dir #'/home/tsm/Code/mmdetection/demo/tutorial_exps/'

