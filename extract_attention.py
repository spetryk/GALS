import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import random
import torch
from torchvision import transforms
import torchvision
import PIL
from CLIP.clip import clip
from tqdm import tqdm

import datasets
from datasets import normalizations
from utils import attention_utils as au


parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str,
                    default='./configs/waterbirds_attention_clip.yaml')
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")

flags  = parser.parse_args()
overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
args      = OmegaConf.merge(cfg, overrides)
args.yaml = flags.config

# reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'

# ***** Set dataset *****
DATASET_KWARGS = {}
if args.DATA.DATASET == 'waterbirds':
    from datasets.waterbirds import Waterbirds as Dataset
    ROOT = os.path.join(args.DATA.ROOT, args.DATA.WATERBIRDS_DIR)
    SAVE_PATH = os.path.join(args.DATA.ROOT, args.DATA.WATERBIRDS_DIR, args.SAVE_FOLDER)
elif args.DATA.DATASET == 'coco_gender':
    from datasets.coco import COCOGender as Dataset
    ROOT = os.path.join(args.DATA.ROOT, 'COCO', 'COCO_gender')
    SAVE_PATH = os.path.join(args.DATA.ROOT,'COCO', 'COCO_gender', args.SAVE_FOLDER)
    DATASET_KWARGS['compute_all'] = True
elif args.DATA.DATASET == 'food_subset':
    from datasets.food import FoodSubset as Dataset
    ROOT = os.path.join(args.DATA.ROOT, 'food-101')
    SAVE_PATH = os.path.join(args.DATA.ROOT, 'food-101', args.SAVE_FOLDER)
else:
    raise NotImplementedError
if not os.path.exists(SAVE_PATH):
    print('>>> Creating folder: {}'.format(SAVE_PATH))
    os.makedirs(SAVE_PATH)
else:
    print('>>> Path exists: {}\n Attention will be overwritten.'.format(SAVE_PATH))

save_config_path = os.path.join(SAVE_PATH, 'extract_attention_config.yaml')
print('>>> Saving config to {}'.format(save_config_path))
with open(save_config_path, 'w') as f:
    OmegaConf.save(config=args, f=save_config_path)

def main(args):
    # Create dataset only to get filenames.
    dataset = Dataset(
        root=args.DATA.ROOT,
        cfg=args,
        transform=None, # Hardcoded: No transform for CLIP
        split='train',
        **DATASET_KWARGS
    )

    # Note: dataset.data must return list of all filenames, regardless of split
    # dataset.labels must return list of corresponding labels.
    # This feature was added later; to make sure it's compatible with all datasets,
    # check them explicitly before adding.
    filenames = dataset.data
    if args.DATA.DATASET == 'waterbirds':
        labels = dataset.labels
        assert len(filenames) == len(labels)
    elif not (args.APPROACH == 'clip' and 'USE_PROMPTS_PER_CLASS' in args and args.USE_PROMPTS_PER_CLASS):
        # Not the setting that needs different prompts per label
        pass
    else:
        raise NotImplementedError

    if args.APPROACH == 'clip':
        print('>>> Loading {} CLIP model'.format(args.MODEL_TYPE))
        model, preprocess = clip.load(args.MODEL_TYPE, device=DEVICE, jit=False)
        preprocess_no_crop = []
        for t in preprocess.transforms:
            if type(t) == torchvision.transforms.transforms.Resize:
                preprocess_no_crop.append(
                    transforms.Resize((224,224), interpolation=PIL.Image.BICUBIC)
                )
            else:
                if type(t) != torchvision.transforms.transforms.CenterCrop:
                    preprocess_no_crop.append(t)
        preprocess = transforms.Compose(preprocess_no_crop)

        if 'USE_PROMPTS_PER_CLASS' in args and args.USE_PROMPTS_PER_CLASS:
            prompt_type = 'per_class'
            prompts = list(args.PROMPTS_PER_CLASS)
            tokenized_text = []
            for cls_prompts in prompts:
                cls_text = clip.tokenize(cls_prompts).to(DEVICE)
                tokenized_text.append(cls_text)
        else:
            prompt_type = 'general'
            prompts = list(args.PROMPTS)
            tokenized_text = clip.tokenize(prompts).to(DEVICE)

        transform = preprocess
    else:
        raise NotImplementedError

    num_visualized = 0

    for i, f in enumerate(tqdm(filenames)):
        tail = f.split(ROOT)[-1]
        tail_path = os.path.join(SAVE_PATH, *tail.split(os.sep)[:-1])
        os.makedirs(tail_path, exist_ok=True)

        plot_vis = (i % 50 == 0) and num_visualized < 30
        if plot_vis:
            num_visualized += 1
        os.makedirs(os.path.join(SAVE_PATH, 'vis'), exist_ok=True)
        save_vis_path = os.path.join(
            os.path.join(
                SAVE_PATH,
                'vis',
                tail.split(os.sep)[-1]
            )
        )

        if prompt_type == 'general':
            text_list = prompts
            clip_text = tokenized_text
        else:
            label = int(labels[i])
            text_list = prompts[label]
            clip_text = tokenized_text[label]

        attention = None
        if args.ATTENTION_TYPE == 'transformer':
            attention = au.transformer_attention(
                model,
                transform,
                f,
                text_list=text_list,
                tokenized_text=clip_text,
                device=DEVICE,
                plot_vis=plot_vis,
                save_vis_path=save_vis_path,
                resize=False
            )
        elif args.ATTENTION_TYPE == 'gradcam':
            attention = au.clip_gcam(
                model,
                transform,
                f,
                text_list=text_list,
                tokenized_text=clip_text,
                layer=args.TARGET_LAYER,
                device=DEVICE,
                plot_vis=plot_vis,
                save_vis_path=save_vis_path,
                resize=False
            )
        else:
            raise NotImplementedError
        assert attention is not None
        save_filepath = os.path.join(SAVE_PATH, tail.strip(os.sep))
        save_filepath = save_filepath.replace('.jpg', '.pth')
        torch.save(attention, save_filepath)


if __name__ == '__main__':
    main(args)




