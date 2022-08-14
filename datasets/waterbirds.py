import torch
from torchvision import transforms
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from PIL import Image

GROUP_NAMES = np.array(['Land_on_Land', 'Land_on_Water', 'Water_on_Land', 'Water_on_Water'])

def get_label_mapping():
    return np.array(['Landbird', 'Waterbird'])

class Waterbirds(torch.utils.data.Dataset):
    def __init__(self, root, cfg, split='train', transform=None):
        self.cfg = cfg
        self.original_root       = os.path.expanduser(root)
        self.transform  = transform
        self.split      = split
        self.root       = os.path.join(self.original_root, cfg.DATA.WATERBIRDS_DIR)
        self.return_seg = True
        self.return_bbox = True
        self.size       = cfg.DATA.SIZE
        self.remove_background = cfg.DATA.REMOVE_BACKGROUND
        self.return_attention = cfg.DATA.ATTENTION_DIR != "NONE"

        print('WATERBIRDS DIR: {}'.format(self.root))

        self.seg_transform = transforms.Compose([
            transforms.Resize((self.size,self.size)),
            transforms.ToTensor(),
        ])

        # metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.root, 'metadata.csv'))

        # Get the y values
        self.labels = self.metadata_df['y'].values
        self.num_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.labels*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.seg_data  =  np.array([os.path.join(root, 'CUB_200_2011/segmentations',
                                                 path.replace('.jpg', '.png')) for path in self.filename_array])

        self.data = np.array([os.path.join(self.root, filename) for filename in self.filename_array])

        if self.return_attention:
            self.attention_data = np.array([os.path.join(self.root, cfg.DATA.ATTENTION_DIR,
                                                path.replace('.jpg', '.pth')) for path in self.filename_array])

        mask = self.split_array == self.split_dict[self.split]
        num_split = np.sum(mask)
        self.indices = np.where(mask)[0]

        self.labels = torch.Tensor(self.labels)
        self.group_array = torch.Tensor(self.group_array)

        # Arrays holding image filenames and labels for just the split, not all data.
        # Useful for detection approach to quickly access labels & filenames
        self.image_filenames     = []
        self.labels_split        = []
        self.group_labels_split  = []
        for idx in self.indices:
            self.image_filenames.append(self.data[idx])
            self.labels_split.append(self.labels[idx])
            self.group_labels_split.append(self.group_array[idx])
        self.image_filenames    = np.array(self.image_filenames)
        self.labels_split       = torch.Tensor(self.labels_split)
        self.group_labels_split = torch.Tensor(self.group_labels_split)

        if self.return_bbox:
            bboxes = pd.read_csv(os.path.join(self.original_root, 'CUB_200_2011', 'bounding_boxes.txt'), header=None)
            self.bbox_coords = np.zeros((self.data.shape[0], 4))
            for i, row in enumerate(bboxes.values):
                coords = row[0].split(' ')[1:]
                coords = np.array(coords).astype(float)
                self.bbox_coords[i] = coords


        print('NUMBER OF SAMPLES WITH LABEL {}: {}'.format(get_label_mapping()[0],
                                                           len(torch.where(self.labels[self.indices] == 0)[0]))
        )
        print('NUMBER OF SAMPLES WITH LABEL {}: {}'.format(get_label_mapping()[1],
                                                           len(torch.where(self.labels[self.indices] == 1)[0]))
        )

        for i in range(len(GROUP_NAMES)):
            print('NUMBER OF SAMPLES WITH GROUP {}: {}'.format(GROUP_NAMES[i],
                                                               len(torch.where(self.group_array[self.indices] == i)[0]))
            )

    def create_subset(self):
        subset_size = self.cfg.DATA.SUBSET_SIZE
        images_per_class = subset_size // 2
        inds = {
            'class_0': torch.where(self.labels[self.indices] == 0)[0],
            'class_1': torch.where(self.labels[self.indices] == 1)[0]
        }

    def get_filenames(self, indices):
        """
        Return list of filenames for requested indices.
        Need to access self.indices to map the requested indices to the right ones in self.data
        """
        filenames = []
        for i in indices:
            new_index = self.indices[i]
            filenames.append(self.data[new_index])
        return filenames


    def __getitem__(self, index):

        original_index = index
        index = self.indices[index]

        path     = self.data[index]
        label    = self.labels[index]
        seg_path = self.seg_data[index]
        group    = self.group_array[index]

        group = torch.Tensor([group])

        img = Image.open(path).convert('RGB')
        if self.return_seg:
            seg = Image.open(seg_path)
        if self.return_bbox:
            bbox = self.bbox_coords[index]
            bbox = np.round(bbox).astype(int)

            arr = np.array(img)
            bbox_im = np.zeros(arr.shape[:2])
            bbox_im[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0]+bbox[2])] = 1
            bbox_im = torch.Tensor(bbox_im).unsqueeze(0).unsqueeze(0)
            bbox_im = F.interpolate(bbox_im, size=(self.size, self.size), mode='bilinear',
                                    align_corners=False)[0]
        else:
            bbox_im  = torch.Tensor([-1]) # So batches correctly

        if self.return_attention:
            att = torch.load(self.attention_data[index])
            att = att['unnormalized_attentions']
            att = F.interpolate(att, size=(self.size, self.size), mode='bilinear',
                                align_corners=False)

        else:
            att  = torch.Tensor([-1]) # So batches correctly

        if self.transform is not None:
            img = self.transform(img)

        if self.return_seg:
            seg = self.seg_transform(seg)
            if self.remove_background:
                img = img * seg
        else:
            seg = torch.Tensor([-1]) # So batches correctly

        return {
            'image_path': path,
            'image': img,
            'label': label,
            'seg':   seg,
            'group': group,
            'bbox': bbox_im,
            'attention': att,
            'index': original_index,
            'split': self.split
        }


    def __len__(self):
        return len(self.indices)


def get_loss_upweights(bias_fraction=0.95, mode='per_class'):
    """
    For weighting training loss for imbalanced classes.

    Returns 1D tensor of length 2, with loss rescaling weights.

    weight w_c for class c in C is calculated as:
    (1 / num_samples_in_class_c) / (max(1/num_samples_in_c) for c in num_classes)

    """
    assert mode in ['per_class', 'per_group']

    # Map bias fraction to per-class and per-group stats.
    training_dataset_stats = {
        0.95: {
            'per_class': [3682, 1113],
            'per_group': [3498, 184, 56, 1057]
        },
        1.0: {
            'per_class': [3694, 1101]
        }
    }
    counts  = training_dataset_stats[bias_fraction][mode]
    counts  = torch.Tensor(counts)
    fracs   = 1 / counts
    weights = fracs / torch.max(fracs)

    return weights





