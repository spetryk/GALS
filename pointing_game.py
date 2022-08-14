import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchray.attribution.grad_cam import grad_cam as tr_gradcam
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import transforms
import PIL
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import os
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

from utils.rise import RISE
from models.resnet import resnet18, resnet50
from models.resnet_abn import resnet50 as resnet50_abn
import utils.general_utils as gu
import utils.attention_utils as au
from datasets import normalizations
import datasets


def load_checkpoint(checkpoint_file, net):
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    state_dict = gu.check_module_state_dict(checkpoint['model_state_dict'])
    net.load_state_dict(state_dict)
    print('Loaded checkpoint {}'.format(checkpoint_file))


def plot(numpy_image, label_float, prob_float, attention, label_mapping=None, title=None, save_path=None):
    if label_mapping is not None:
        cls = label_mapping[int(label_float)]
    else:
        cls = 'Class {}'.format(int(label_float))

    fig, ax = plt.subplots(1,2, figsize=(5,6))
    ax[0].imshow(numpy_image)
    ax[0].axis('off')

    ax[1].imshow(numpy_image)
    ax[1].imshow(attention, alpha=0.4, cmap='jet')
    ax[1].axis('off')

    temp_title = ''
    if title is not None:
        temp_title = title + '\n'
    temp_title += '{}: {:.3f}'.format(cls, prob_float)

    ax[1].set_title(temp_title, fontsize=14)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    return


def normalize(sal):
    # sal = tensor of shape 1,1,H,W
    B, C, H, W = sal.shape
    sal = sal.view(B, -1)
    sal_max = sal.max(dim=1, keepdim=True)[0]
    sal_max[torch.where(sal_max == 0)] = 1. # prevent divide by 0
    sal -= sal.min(dim=1, keepdim=True)[0]
    sal /= sal_max
    sal = sal.view(B, C, H, W)
    return sal


class ActivationNet(nn.Module):
    def __init__(self, net, num_classifier_classes=1):
        super(ActivationNet, self).__init__()
        self.net = net
        self.num_classifier_classes = num_classifier_classes

        if num_classifier_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        logits = self.net(inputs)
        probs, _, _ = gu.calc_preds(logits, self.activation, self.num_classifier_classes)
        return probs


class ActivationNetABN(ActivationNet):
    def __init__(self, net, num_classifier_classes=1):
        super(ActivationNetABN, self).__init__(net, num_classifier_classes)

    def forward(self, inputs):
        _, logits, _ = self.net(inputs, provided_att=None)
        probs, _, _ = gu.calc_preds(logits, self.activation, self.num_classifier_classes)
        return probs



parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str,
                    default='./configs/waterbirds_generic.yaml')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Evaluate checkpoint file on test set')
parser.add_argument('--num_masks', type=int, default=2000,
                    help='Number of masks to use for RISE')
parser.add_argument('--generate_new', action='store_true',
                    help='Generate new masks for RISE')
parser.add_argument('--num_classifier_classes', type=int, default=1,
                    help='Number of classifier outputs')
parser.add_argument('--num_classes', type=int, default=2,
                    help='Number of classes')
parser.add_argument('--attention_type', type=str, default='rise',
                    help='Attention computation: rise or gradcam')

parser.add_argument(
    'overrides', nargs='*',
    help="Any key=value arguments to override config values (use dots for.nested=overrides)"
)

flags  = parser.parse_args()
overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base_cfg  = OmegaConf.load('configs/base.yaml')
args      = OmegaConf.merge(base_cfg, cfg, overrides)
args.yaml = flags.config
args.checkpoint = flags.checkpoint
args.generate_new = flags.generate_new
args.num_masks = flags.num_masks
args.num_classifier_classes = flags.num_classifier_classes
args.num_classes = flags.num_classes

# reproducibility
seed = args.SEED
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


size = args.DATA.SIZE
model_type = args.EXP.MODEL

num_classifier_classes = args.num_classifier_classes
num_classes = args.num_classes
attention_type = flags.attention_type
assert attention_type in ['rise', 'gradcam']
print('RUNNING WITH ATTENTION TYPE: {}'.format(attention_type.upper()))

if model_type == 'resnet50':
    net = resnet50(pretrained=False, num_classes = num_classifier_classes)
elif model_type == 'resnet50_abn':
    net = resnet50_abn(pretrained=False, num_classes=num_classifier_classes)
else:
    net = resnet18(pretrained=False, num_classes = num_classifier_classes)

checkpoint = args.checkpoint


# Prevent OOM when loading big state dict
#device = 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'
device = 'cuda'
temp = os.environ['CUDA_VISIBLE_DEVICES']
os.environ['CUDA_VISIBLE_DEVICES'] = ''
load_checkpoint(checkpoint, net)
os.environ['CUDA_VISIBLE_DEVICES'] = temp


# Add activation
if model_type == 'resnet50_abn':
    net = ActivationNetABN(net, num_classifier_classes)
else:
    net = ActivationNet(net, num_classifier_classes)

net.to(device)
net.eval()


# ***** Set dataset *****
dataset_type = args.DATA.DATASET
print(dataset_type)
if dataset_type == 'waterbirds':
    from datasets.waterbirds import Waterbirds as Dataset
    label_mapping = datasets.waterbirds.get_label_mapping()
elif dataset_type == 'coco_gender':
    from datasets.coco import COCOGender as Dataset
else:
    raise NotImplementedError

mean, std = normalizations.normalizations['imagenet']['mean'], \
            normalizations.normalizations['imagenet']['std']
transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

dataset = Dataset(
    root='./data',
    cfg=args,
    transform=transform,
    split='test'
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=args.DATA.NUM_WORKERS,
    shuffle=False
)

if attention_type == 'rise':
    # RISE setup
    rise = RISE(net,
                (size, size),
                num_classes=num_classes,
                gpu_batch=16,
                p1=0.1)
    # Generate masks for RISE or use the saved ones.
    maskspath = 'masks.npy'
    generate_new = args.generate_new
    rise_N = args.num_masks
    rise_s = 8
    if generate_new or not os.path.isfile(maskspath):
        rise.generate_masks(N=rise_N, s=rise_s, device=device, savepath=maskspath)
    else:
        rise.load_masks(maskspath, device)
        print('Masks are loaded.')
    torch.set_grad_enabled(False)

net.eval()


# PG stats
num_correct = 0.
num_total   = 0.

class_corrects = np.zeros(num_classes)
class_totals = np.zeros(num_classes)

for i, batch in enumerate(tqdm(dataloader)):
    inputs, labels = batch['image'].to(device), batch['label'].to(device)
    seg = batch['seg']
    seg = seg[0][0].numpy() # 224 x 224

    img_path = batch['image_path'][0]
    pil_image =  transforms.Resize((size, size))(Image.open(img_path).convert('RGB'))
    numpy_image = np.array(pil_image)

    with torch.no_grad():
        probs  = net(inputs) # 1 x num_classes
        prob_float = float(probs[0][labels[0].long()].cpu())
        label_float = float(labels[0])

    if attention_type == 'rise':
        saliencies = rise(inputs)
        sal = saliencies[labels[0].long()].cpu().numpy()  # 224 x 224
    else:
        saliencies = tr_gradcam(net, inputs, labels[0].long(), saliency_layer='net.layer4.2.relu')
        saliencies = F.interpolate(
                    saliencies, (224,224), mode="bilinear", align_corners=False
        )
        saliencies = normalize(saliencies)
        sal = saliencies[0][0].detach().cpu().numpy()


    if np.max(sal) == 0:
        found = False
    else:
        sal_max_inds = np.where(sal == np.max(sal))
        found = False
        for x,y in zip(sal_max_inds[0], sal_max_inds[1]):
            # Count as correct in PG if any of the maxes are inside segmentation
            # (accounts for case if saliency has max value in multiple places)
            if not found:
                if seg[x,y] == 1:
                    num_correct += 1
                    found = True
                    class_corrects[int(label_float)] += 1

    num_total += 1
    class_totals[int(label_float)] += 1

    # plot(numpy_image,
    #      label_float,
    #      prob_float,
    #      sal,
    #      label_mapping=label_mapping,
    #      title='PG {}'.format('correct' if found else 'incorrect'),
    #      save_path='test_gcam_{}.png'.format(i)
    # )


print('POINTING GAME RESULTS FOR CHECKPOINT: {}'.format(checkpoint))
print('NUM CORRECT: {}, NUM TOTAL: {}'.format(num_correct, num_total))
print('PG AVERAGE ACC: {}'.format(num_correct / num_total))

print('PG CLASS ACCS:')
print(class_corrects)
print(class_totals)
for i in range(len(class_corrects)):
    print('LABEL {}: {}'.format(i, class_corrects[i]/class_totals[i]))



