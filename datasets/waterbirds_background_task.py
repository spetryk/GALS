import torch
from torchvision import transforms
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from PIL import Image

from datasets.waterbirds import Waterbirds

GROUP_NAMES = np.array(['Landbird_on_Land', 'Landbird_on_Water', 'Waterbird_on_Land', 'Waterbird_on_Water'])

def get_label_mapping():
    return np.array(['Land', 'Water'])

class WaterbirdsBackgroundTask(Waterbirds):
    """
    Same data as Waterbirds, except labels are changed to Water & Land background,
    with the type of bird as the confounding variable.
    """
    def __init__(self, root, cfg, split='train', transform=None, size=None):
        super().__init__(root, cfg, split=split, transform=transform, size=size)


        # Use group labels to get the background type
        background_labels = []
        for idx in range(len(self.group_array)):
            group_label = self.group_array[idx]
            # water background is label 1, land is label 0.
            is_water = (group_label == 1) or (group_label == 3)
            background_labels.append(int(is_water))

        self.labels = torch.Tensor(background_labels)

        labels_split = []
        for idx in self.indices:
            labels_split.append(self.labels[idx])
        self.labels_split = labels_split

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
            'per_class': [3554, 1241],
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





