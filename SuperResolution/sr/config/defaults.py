# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN

# NOTE: given the new config system
# (https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html),
# we will stop adding new functionalities to default CfgNode.

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DEBUG = False
_C.OUTPUT_DIR = ""
_C.SEED = 3407
_C.DETERMINISTIC = True
_C.RESUME_CKPT_PATH = None
_C.PRETRAINED_CKPT_PATH = None

# DATASETS
_C.DATASETS = CN()
_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.PATHS = []
_C.DATASETS.TRAIN.ITERATION = 1000
_C.DATASETS.TRAIN.CROP_SIZE = [(164, 246), (256, 384), (384, 576), (512, 768)]
_C.DATASETS.TRAIN.AUGMENT = True

_C.DATASETS.VAL = CN()
_C.DATASETS.VAL.PATH = ""

_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.PATH = ""

# DATALOADER
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = [10, 7, 3, 2, 1, 1]
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.PIN_MEMORY = False

# MODEL
_C.MODEL = CN()
_C.MODEL.SR = CN()
_C.MODEL.SR.UP_SCALE = 2
_C.MODEL.SR.NAME = "SR_MixS2"
_C.MODEL.SR.SPECTRAL_BRANCH = True
_C.MODEL.SR.SPATIAL_BRANCH = True
_C.MODEL.SR.SPECTRAL_INTERACTION = True
_C.MODEL.SR.SPATIAL_INTERACTION = True
_C.MODEL.SR.IN_DIM = 3
_C.MODEL.SR.OUT_DIM = 3
_C.MODEL.SR.DIM = 56
_C.MODEL.SR.NUM_HEADS = 2
_C.MODEL.SR.NUM_BLOCKS = 4
_C.MODEL.SR.DW_EXPAND = 1
_C.MODEL.SR.FFN_NAME = "Gated_Dconv_FeedForward"
_C.MODEL.SR.FFN_EXPAND = 2.66
_C.MODEL.SR.BIAS = False
_C.MODEL.SR.LAYERNORM_TYPE = "BiasFree"
_C.MODEL.SR.ACT_FN_NAME = "gelu"

# DISCRIMINATOR
_C.MODEL.DISCRIMINATOR = CN()
_C.MODEL.DISCRIMINATOR.NAME = "Discriminator"
_C.MODEL.DISCRIMINATOR.IN_DIM = 3
_C.MODEL.DISCRIMINATOR.LR = 1e-4
# EMA
_C.MODEL.EMA = CN()
_C.MODEL.EMA.ENABLE = True
_C.MODEL.EMA.DECAY = 0.999

# LOSS
_C.LOSS = CN()
_C.LOSS.L1_LOSS = True
_C.LOSS.TV_LOSS = True
_C.LOSS.SSIM_LOSS = True
_C.LOSS.VGG_LOSS = True
_C.LOSS.GAN_LOSS = True


# OPTIMIZER
_C.OPTIMIZER = CN()
_C.OPTIMIZER.MAX_EPOCH = 300
_C.OPTIMIZER.LR = 2e-4
_C.OPTIMIZER.GRAD_CLIP = True


