import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import random
import torch
from torchvision import transforms
import pandas as pd

import datasets
from datasets import normalizations

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str,
                    default='./configs/waterbirds_generic.yaml')
parser.add_argument('--dryrun', action='store_true',
                    help='Use flag to prevent logging to wandb server (keeps local instead)')
parser.add_argument('--test_checkpoint', type=str, default=None,
                    help='Evaluate checkpoint file on test set')
parser.add_argument('--name', type=str, default=None, help='name for wandb run')

parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")

flags  = parser.parse_args()
overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base_cfg  = OmegaConf.load('configs/base.yaml')
args      = OmegaConf.merge(base_cfg, cfg, overrides)
args.yaml = flags.config


# reproducibility
seed = args.SEED
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ***** Set approach *****

if args.EXP.APPROACH == 'generic':
    from approaches.generic_cnn import GenericCNN as Approach
elif args.EXP.APPROACH == 'abn':
    from approaches.abn import ABN as Approach
elif args.EXP.APPROACH == 'coco_gender':
    from approaches.coco_gender import COCOGenderCNN as Approach
elif args.EXP.APPROACH == 'coco_abn':
    from approaches.coco_abn import COCOABN as Approach
else:
    raise NotImplementedError

DEVICE = 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'

# ***** Set dataset *****
if args.DATA.DATASET == 'waterbirds':
    from datasets.waterbirds import Waterbirds as Dataset
elif args.DATA.DATASET == 'waterbirds_background':
    from datasets.waterbirds_background_task import WaterbirdsBackgroundTask as Dataset
elif args.DATA.DATASET == 'coco_gender':
    from datasets.coco import COCOGender as Dataset
elif args.DATA.DATASET == 'food_subset':
    from datasets.food import FoodSubset as Dataset
else:
    raise NotImplementedError


# ***** Setup logging *****
wandb_dir = os.path.join('.', 'wandb', args.DATA.DATASET)
os.makedirs(wandb_dir, exist_ok=True)
os.environ['WANDB_DIR'] = wandb_dir
if flags.dryrun:
    os.environ['WANDB_MODE'] = 'dryrun'
else:
    os.environ['WANDB_MODE'] = 'run'
args.name = flags.name

# if flags.name is not None:
#     os.environ['WANDB_RUN_ID'] = flags.name

# Switch to test setting if test checkpoint file given
args.test_checkpoint = flags.test_checkpoint


def main(args):
    print(OmegaConf.to_yaml(args))

    # Transforms
    mean, std = normalizations.normalizations[args.DATA.NORMALIZATION]['mean'], \
                normalizations.normalizations[args.DATA.NORMALIZATION]['std']
    transform = transforms.Compose([
        transforms.Resize((args.DATA.SIZE, args.DATA.SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if args.test_checkpoint is not None:
        # Create approach w/o train & val datasets, and go directly to test
        approach = Approach(args, [None, None])
        test_metrics = test(args, transform, approach, mean, std)
        return

    # Data
    train_dataset = Dataset(root=args.DATA.ROOT,
                            cfg=args,
                            transform=transform,
                            split='train')
    val_dataset = Dataset(root=args.DATA.ROOT,
                          cfg=args,
                          transform=transform,
                          split='val')

    print('NUM TRAIN: {}\n'.format(len(train_dataset)))
    print('NUM VAL:   {}\n'.format(len(val_dataset)))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.DATA.BATCH_SIZE,
                                                   num_workers=args.DATA.NUM_WORKERS,
                                                   shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.DATA.BATCH_SIZE,
                                                 num_workers=args.DATA.NUM_WORKERS,
                                                 shuffle=False)

    # If the wandb run name is specified, and you're running >1 trial,
    # append the trial number to the end of the name so each run gets a unique name.
    # For this, we save the original run name in base_run_name
    if args.name is not None:
        base_run_name = args.name
    else:
        base_run_name = None
    running_metrics = []
    for trial_num in range(args.EXP.NUM_TRIALS):
        if args.EXP.NUM_TRIALS > 1 and base_run_name is not None:
            args.name = '{}_trial_{}'.format(base_run_name, trial_num)
        approach = Approach(args, [train_dataloader, val_dataloader])
        if args.test_checkpoint is None or trial_num > 0:
            # Approach may set test_checkpoint field to run test after training finishes
            approach.train()
            args.test_checkpoint = approach.test_checkpoint

        test_metrics = test(args, transform, approach, mean, std)
        args.test_checkpoint = None # Set back to None for rest of trials
        del approach
        running_metrics.append(test_metrics)

        if args.LOGGING.SAVE_STATS_PATH is not None:
            # Append stats to CSV file containing stats for same run, but different trials.
            values = []
            cols = []
            for k,v in test_metrics.items():
                cols.append(k)
                values.append(v.avg)
            if not os.path.exists(args.LOGGING.SAVE_STATS_PATH):
                mode = 'w' # Create a new file
                header = True
            else:
                mode = 'a' # Append to existing file
                header = False
            df = pd.DataFrame([values], columns=cols)
            df.to_csv(args.LOGGING.SAVE_STATS_PATH, mode=mode, header=header)

    if args.EXP.NUM_TRIALS > 1:
        print('*********************************************************************')
        print('******************* AVERAGE STATS OVER {} TRIALS ********************'.format(args.EXP.NUM_TRIALS))
        print('*********************************************************************')
        assert len(running_metrics) == args.EXP.NUM_TRIALS
        keys = running_metrics[0].keys()
        for k in keys:
            vals = np.array([m[k].avg for m in running_metrics])
            mean = np.mean(vals)
            std = np.std(vals)
            if 'acc' in k:
                mean *= 100
                std *= 100
            print('{}: {} +/- {}'.format(k, mean, std))


def test(args, transform, approach, mean, std):
    print('>>> EVALUATING ON TEST SET')
    test_dataset = Dataset(root=args.DATA.ROOT,
                           cfg=args,
                           transform=transform,
                           split='test')

    print('NUM TEST:   {}\n'.format(len(test_dataset)))

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  num_workers=args.DATA.NUM_WORKERS,
                                                  shuffle=False)
    metrics = approach.test(test_dataloader, args.test_checkpoint)
    return metrics


if __name__ == '__main__':
    main(args)

