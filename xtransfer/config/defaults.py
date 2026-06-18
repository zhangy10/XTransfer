import os

from yacs.config import CfgNode as CN

# repo root = three levels up from this file (xtransfer/config/defaults.py)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

###########################
# Config definition
#
# This file is the SCHEMA: it declares every legal config key. Defaults below
# already correspond to the XTransfer "Our-Single" method on HHAR, so the repo
# runs out of the box. A run's authoritative config is `configs/hhar_single.yaml`
# (merged on top of these defaults); only per-run knobs (shot / fold) come from
# the CLI. No config key may be created at runtime — declare it here.
###########################

_C = CN()

# Output directory (logs, log_dict.pkl, saved models)
_C.OUTPUT_DIR = os.path.join(_REPO, "output")
_C.OUTPUT_DATA_DIR = os.path.join(_REPO, "output")
# Global random seed (fixed for reproducibility)
_C.SEED = 5
_C.USE_CUDA = True

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.DATA_NAME = 'HHAR'
_C.DATALOADER.NUM_SHOTS = 5
_C.DATALOADER.NUM_WORKERS = 0
# few-shot episode sampling mode: 'd' | 'i' | 'p'
_C.DATALOADER.MODE = 'p'
# input transform: 'Raw2D' (2D backbones) | 'Raw1D' | 'Raw1D_resize'
_C.DATALOADER.TRANS_METHOD = 'Raw2D'
_C.DATALOADER.SEED = None
_C.DATALOADER.RESIZE = 224

###########################
# RSR / XTransfer method
###########################
_C.RSR = CN()
# few-shot test classifier on latent features
_C.RSR.TEST_METHOD = 'KNN'
# max layer depth examined by layer-wise search
_C.RSR.BREAK_DEPTH = 9
# fit pipeline: 'repair' = SRR (Splice-Repair-Removal, includes Pruner)
_C.RSR.FIT_MODE = 'repair'
# repair loss mode
_C.RSR.LOSS_MODE = 'repair'
# normalisation mode for connectors ('IS' = instance-standardisation)
_C.RSR.NORM_MODE = 'IS'
# anchor PCA components (reduced-orthogonal feature space)
_C.RSR.PCA_COMPONENT = 2
# LWS search depth (paper default 3)
_C.RSR.SEARCH_DEPTH = 3
_C.RSR.TEST_ACC = True
_C.RSR.REGRESSION = False
_C.RSR.SBP = False
# 1D backbone path (False for the 2D ResNet18 source)
_C.RSR.Conv1D = False
# final linear fine-tune after layer recombining
_C.RSR.FINETUNE = True
_C.RSR.END_CNN = False
# pick the best-validation recombined model
_C.RSR.BEST_VALIDATION = True
# layer-wise search switches
_C.RSR.RESOURCE_SWITCH = True
_C.RSR.RESIZER_SWITCH = True
_C.RSR.ROLLBACK_SWITCH = True
_C.RSR.BEST_ANCHOR = False
_C.RSR.POOL_SEED = False
# pre-search check / quick-check acceleration
_C.RSR.QUICK_CHECK = True
# PCA-based channel removal range
_C.RSR.MAX_PRUNE_RANGE = 1.0
# device resource budget (None = unconstrained)
_C.RSR.MAX_RESOURCE = None
# initial candidate pick range per search pool
_C.RSR.INIT_PICK_RANGE = 0.5

###########################
# Model Pool (source pre-trained models)
###########################
_C.MODEL_POOL = CN()
_C.MODEL_POOL.NAMES = ('miniImageNet',)
_C.MODEL_POOL.DIR = os.path.join(_REPO, 'pre-trained_weights')
_C.MODEL_POOL.MATCH_JSON = os.path.join(_REPO, 'xtransfer', 'config', 'match.json')
_C.MODEL_POOL.STRUC_JSON = os.path.join(_REPO, 'xtransfer', 'config', 'struc.json')
_C.MODEL_POOL.NAMED_STRUC_JSON = os.path.join(_REPO, 'xtransfer', 'config', 'named_struc.json')
_C.MODEL_POOL.MODEL_NAME = 'model.pth.tar'
_C.MODEL_POOL.ANCHOR_NAME = 'anchor_activation_mmc.pkl'

###########################
# Generative transfer module (connector training)
###########################
_C.TRAN = CN()
_C.TRAN.PRINT_FREQ = 50
_C.TRAN.ANCHOR_SELECT_MODE = 'best_pca'
_C.TRAN.PCA_MODE = 'all'
_C.TRAN.NUM_EPISODE = 200
_C.TRAN.ROT_NUM_EPISODE = 20
_C.TRAN.LOSS = 'Npair&PositiveAnchor'
_C.TRAN.RM = True

