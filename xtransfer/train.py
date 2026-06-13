import os

# deterministic cuBLAS (must be set before torch initialises CUDA)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time
import warnings

import torch
from thop import clever_format, profile

from utils import setup_logger, set_random_seed, Notes, RSRLogger
from xtransfer.core import MatchingNet
from xtransfer.config import get_cfg_default
from xtransfer.target_datasets import create_dataloader
from utils.ResourceProfile import RP, stop_RP
from xtransfer.tools import replace_relu
from xtransfer.trans import Finetuner

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# default experiment config (single source of truth for a run)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(_REPO, 'configs', 'hhar_single.yaml')


def get_fit_schema(fit_mode):
    """Map a fit-mode name to its stage schema.

    'repair' is the XTransfer SRR pipeline (Splice-Repair-Removal, with Pruner).
    The others are ablations kept for reference; the default entry uses 'repair'.
    """
    schemas = {
        'repair':      {'prehead': 'Repair', 'pruner': 'Pruner', 'afterhead': None, 'trans': None},
        'repair_noP':  {'prehead': 'Repair', 'pruner': None,     'afterhead': None, 'trans': None},
        'bi':          {'prehead': 'Bi_Contrast', 'afterhead': 'Bi_Contrast', 'trans': None},
        'before':      {'prehead': 'Contrast', 'afterhead': None, 'trans': None},
        'native':      {'prehead': 'Native', 'afterhead': None, 'trans': None},
        'og':          {'prehead': None, 'afterhead': None, 'trans': None},
    }
    return schemas[fit_mode]


def get_currect_size(model_name):
    resnet18 = ['miniImageNet', 'miniDomainNet', 'caltech', 'office31', 'officeHome', 'VoxCeleb']
    resnet10 = ['CIFAR', 'CUB', 'DTD', 'Omniglot', 'QuickDraw']
    conv4 = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    conv41d = ['MHEALTH', 'OPPORTUNITY', 'PAMAP2', 'sEMG', 'UniMiB']
    bert = ['News_bert']
    resnet181d = ['News']
    if model_name in resnet18:
        return 224, 'resnet18'
    elif model_name in resnet10:
        return 84, 'resnet10'
    elif model_name in conv4:
        return 32, 'conv4'
    elif model_name in conv41d:
        return 100, 'conv41d'
    elif model_name in resnet181d:
        return 100, 'resnet181d'
    elif model_name in bert:
        return 512, 'bert'
    else:
        raise ValueError('This model is not supported yet!')


def build_cfg(model_name, dataset, n_shot, config_file=DEFAULT_CONFIG):
    """Build the run config: defaults <- yaml <- per-run arguments.

    Fixed method hyper-parameters live in `config_file`; only the per-run knobs
    (source models, target dataset, shot) and the backbone-derived input size
    are set here from arguments.
    """
    cfg = get_cfg_default()
    if config_file and os.path.isfile(config_file):
        cfg.merge_from_file(config_file)

    cfg.MODEL_POOL.NAMES = model_name
    cfg.DATALOADER.DATA_NAME = dataset
    cfg.DATALOADER.NUM_SHOTS = n_shot

    # input size / transform are determined by the source backbone
    cfg.DATALOADER.RESIZE, backbone_model = get_currect_size(model_name[0])
    if backbone_model in ['resnet10', 'conv4']:
        cfg.DATALOADER.TRANS_METHOD = 'Raw1D'
    elif backbone_model in ['conv41d', 'resnet181d', 'bert']:
        cfg.DATALOADER.TRANS_METHOD = 'Raw1D_resize'
    else:
        cfg.DATALOADER.TRANS_METHOD = 'Raw2D'
        torch.use_deterministic_algorithms(True)
    return cfg, backbone_model


def main(model_name, dataset, epo_id=0, n_shot=5, config_file=DEFAULT_CONFIG):
    multi_or_single = 'multi' if len(model_name) > 1 else ('single-' + model_name[0])
    cfg, backbone_model = build_cfg(model_name, dataset, n_shot, config_file)

    output_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "{}-{}_{}PCA_{}Mode_{}Shots_{}_{}_PCA".format(
            cfg.DATALOADER.DATA_NAME, cfg.DATALOADER.TRANS_METHOD, cfg.RSR.PCA_COMPONENT,
            cfg.DATALOADER.MODE, cfg.DATALOADER.NUM_SHOTS, multi_or_single, backbone_model))
    output_dir = os.path.join(output_dir, '{}'.format(epo_id))

    setup_logger(output_dir)
    tlogger = RSRLogger(output_dir)
    print('Data Configuration:')
    print(cfg.DATALOADER)

    RP(useGPU=True, filename=output_dir, interval=1)

    fit_schema = get_fit_schema(cfg.RSR.FIT_MODE)
    train, val, test, users = create_dataloader(
        data_name=cfg.DATALOADER.DATA_NAME, resize=cfg.DATALOADER.RESIZE, n_shot=cfg.DATALOADER.NUM_SHOTS,
        trans_method=cfg.DATALOADER.TRANS_METHOD, mode=cfg.DATALOADER.MODE, num_workers=cfg.DATALOADER.NUM_WORKERS,
        seed=cfg.DATALOADER.SEED, return_validation=True, epo_idx=epo_id,
        regression=cfg.RSR.REGRESSION, sbp=cfg.RSR.SBP, users=None)

    set_random_seed(seed=cfg.SEED)

    stime = time.time()
    mNet = MatchingNet(cfg, trainloader=train, testloader=test, valloader=val, fit_schema=fit_schema, logger=tlogger)
    mNet.fit()

    total_time = time.time() - stime
    print('Total search time is {:.3f}'.format(total_time))
    Notes.write('Total search time is {:.3f}'.format(total_time))
    tlogger.write('time', total_time)
    tlogger.write('users', users)
    mNet.backbone.eval()
    model = mNet.backbone

    tr = iter(val)
    tr_x, tr_y = next(tr)
    single_input = tr_x[:1]
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = single_input.device
    single_input = single_input.to(model_device)
    macs, params = profile(model, inputs=(single_input,), verbose=False)
    flops = macs * 2
    formatted_macs, formatted_flops, formatted_params = clever_format([macs, flops, params], "%.3f")
    print('Single input shape is {}'.format(tuple(single_input.shape)))
    print('MACs is {}, FLOPs is {}, Params is {}'.format(formatted_macs, formatted_flops, formatted_params))

    # final linear fine-tune after layer recombining
    if cfg.RSR.FINETUNE:
        torch.use_deterministic_algorithms(False)
        tr = iter(val)
        tr_x, tr_y = next(tr)

        finetuner = Finetuner(x=tr_x, y=tr_y, model=model, step=100, backnet=mNet, logger=tlogger,
                              regression=cfg.RSR.REGRESSION, is_1d=cfg.RSR.Conv1D)
        finetuner.optimize_params()
        linear = finetuner.get_linear()
        torch.save(linear, os.path.join(output_dir, 'linear.pt'))

    tlogger.close()
    stop_RP()

    print(model)
    Notes.write(str(model))
    Notes.save_to_file(os.path.join(output_dir, 'selected_log.txt'))

    mNet.backbone.eval()
    model = mNet.backbone
    replace_relu(model)
    torch.save(model, os.path.join(output_dir, 'model.pt'))

    return output_dir
