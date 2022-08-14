import torch
import torch.distributed as dist
from torchvision import transforms
import numpy as np
import wandb
import tempfile
import pickle
import os
from omegaconf import OmegaConf
import omegaconf
import sys
import pdb

def freeze_weights(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def unfreeze_weights(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True


def freeze_bn(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d) or \
           isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            module.eval()


def get_sample_dataloader(dataset, num_samples, batch_size, num_workers, collate_fn):
    """
    Create dataloader with only num_samples random samples from dataset.
    Used for logging the same samples acoss iterations to directly see progress.
    """
    indices = np.random.choice(np.arange(len(dataset)), size=num_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )

def convert_to_numpy(images, unnormalize):
    """ Convert input images Tensor to numpy HxWxC, RGB channel order.
    unnormalize is torchvision.transforms transform that unnormalizes tensor
    Returns list of numpy arrays
    """
    ims = [unnormalize(im) for im in images]
    ims = [im.cpu().numpy() for im in ims]
    ims = [(np.moveaxis(im, 0, -1) * 255.).astype(np.uint8) for im in ims]
    return ims

def create_unnormalize_transform(mean, std):
    unmean = -np.array(mean) / np.array(std)
    unstd  = 1. / np.array(std)
    return transforms.Normalize(unmean, unstd)

def check_module_state_dict(original_state_dict, force_remove_module=False):
    """ Saved: state_dict """
    sample_layer = [k for k in original_state_dict.keys()][0]
    state_dict = {}
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if (num_gpus < 2 and 'module' in sample_layer) \
       or force_remove_module:
        for key, value in original_state_dict.items():
            # remove module. from start of layer names
            state_dict[key[7:]] = value
    elif num_gpus > 1 and 'module' not in sample_layer:
        for key, value in original_state_dict.items():
            # add module. to beginning of layer names
            state_dict['module.{}'.format(key)] = value
    else:
        state_dict = original_state_dict
    return state_dict


def init_wandb(config):
    flattened_config = flatten_config(config)
    if config.name is not None:
        os.environ['WANDB_RUN_ID'] = config.name
    project_dataset = config.DATA.DATASET
    wandb.init(
        #project='vl-attention-{}'.format(project_dataset),
        project='vl-attention-release',
        resume='never',
        config=flattened_config
    )
    print('WANDB DIR: {}'.format(wandb.run.dir))
    for k,v in flattened_config.items():
        print(k,v)
    filename = os.path.join(wandb.run.dir, 'omegaconf_file.yaml')
    print(filename)
    with open(filename, 'w') as f:
        OmegaConf.save(config=config, f=filename)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        wandb.config.update({'num_gpus': torch.cuda.device_count()})
        wandb.config.update({'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']})


def calc_preds(logits, activation, num_classifier_classes, enforce_binary=False):
    """
    enforce_binary: only compute preds for classes 0 and 1, when
    original classification problem has more than 2 classes

    probs:     B x num_classes
    max_probs: B x 1
    preds:     B x 1

    """
    if enforce_binary and logits.shape[1] > 2:
        logits = logits[:,:2]

    probs = activation(logits)
    if num_classifier_classes == 1:
        class1_probs = 1 - probs
        probs = torch.cat((class1_probs, probs), dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    return probs, max_probs, preds


def flatten_config(dic, running_key=None, flattened_dict={}):
    for key, value in dic.items():
        if running_key is None:
            running_key_temp = key
        else:
            running_key_temp = '{}.{}'.format(running_key, key)
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            flatten_config(value, running_key_temp)
        else:
            flattened_dict[running_key_temp] = value
    return flattened_dict


def del_prev_best_model_file(directory):
    """
    Delete file holding previous best model. Instead of having static filename for best model,
    we save the metric in the filename for convenience in understanding the run.
    """
    prev_best = list(filter(lambda f: 'best' in f, os.listdir(directory)))
    print('prev best list: {}'.format(prev_best))
    if len(prev_best) == 1:
        os.system('rm {}'.format(os.path.join(directory, prev_best[0])))
    elif len(prev_best) > 1:
        print(
            'EXCEPTION: Only one saved model file should contain "best" at one time.' \
            'List of all model files found: {}'.format(prev_best)
        )
        raise Exception


def logprint(log=True, message=None):
    """
    For use with distributed training; only print if called by process with local rank 0.
    Avoids the need to check this condition each time.

    Set log to be 'if (not distributed) or (process has local rank 0)'
    Use partial function application:
      logprint = partial(general_utils.logprint, log = log)
    Then call in code as just logprint(message_to_print)
    """
    if log:
        print(message)
        return
    else:
        return


def aggregate_metrics(metrics, local_rank):
    """
    For distributed training, does all-reduce to sync metrics on all GPUs
    metrics: dictionary [Str -> AverageMeter]
             maps metric name to AverageMeter keeping track of the metric.
    Returns:
    aggregated: dictionary [Str -> AverageMeter]
                Calling aggregated[metric_name].avg returns average value of metric
                across all GPUs.
    """
    aggregated = {}
    for metric_name, metric in metrics.items():
        metric_tensor = torch.tensor([metric.sum]).to(torch.device(local_rank))
        count_tensor  = torch.tensor([metric.count]).to(torch.device(local_rank))
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor,  op=dist.ReduceOp.SUM)
        if count_tensor[0] == 0:
            count_tensor[0] = 1 # prevent divide by 0
        reduced_metric = metric_tensor.item() / count_tensor.item()
        aggregated[metric_name] = AverageMeter()
        aggregated[metric_name].update(reduced_metric, n=1)
    return aggregated


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
