"""
Credit to:
Vitali Petsiuk, Abir Das, Kate Saenko
RISE: Randomized Input Sampling for Explanation of Black-box Models
BMVC 2018

https://github.com/eclique/RISE
"""

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm


class RISE(nn.Module):
    def __init__(self, model, input_size, num_classes, gpu_batch=16, p1=0.1):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.num_classes = num_classes
        self.gpu_batch = gpu_batch
        self.p1 = p1

    def generate_masks(self, N, s, device, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < self.p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(
                grid[i], up_size, order=1, mode='reflect',
                anti_aliasing=False
            )[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.to(device)
        self.N = N

    def load_masks(self, filepath, device):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().to(device)
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        #p = nn.Softmax(dim=1)(self.model(stack)) in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))

        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal


def explain_all(images, labels, explainer):
    total = len(labels)
    probs, classpreds = torch.sort(nn.Softmax(1)(explainer.model(images)), dim=1, descending=True)
    explanations = []
    for img in images:
        explanations.append(explainer(torch.unsqueeze(img, 0)))
    return explanations, probs, classpreds
