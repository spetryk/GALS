"""
Credit to:
Kazuto Nakashima
http://kazuto1011.github.io
https://github.com/kazuto1011/grad-cam-pytorch
"""

from collections import OrderedDict, Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import collections

class _BaseWrapper(object):
    """
    Please modify forward() and backward() according to your task.
    """

    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image, text):
        # simple classification
        self.model.zero_grad()
        self.logits, _ = self.model(image, text)
        self.probs = self.logits.softmax(dim=-1).detach().cpu().numpy()[0]
        return self.probs

    def backward(self, ids):
        """
        Class-specific backpropagation

        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """
        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_total = OrderedDict()
        self.grad_total = OrderedDict()
        self.fmap_pool = collections.defaultdict(dict)
        self.grad_pool = collections.defaultdict(dict)
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps for layers with parameters
                self.fmap_total[key] = output.detach()
            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps for layers with parameters
                self.grad_total[key] = grad_out[0].detach()
            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.visual.named_modules():
            if (self.candidate_layers is None or name in self.candidate_layers):
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def reset(self):
        del self.fmap_total
        del self.grad_total
        del self.fmap_pool
        del self.grad_pool

        self.fmap_total = OrderedDict()
        self.grad_total = OrderedDict()
        self.fmap_pool = collections.defaultdict(dict)
        self.grad_pool = collections.defaultdict(dict)

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image, text):
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(image, text)

    def generate(self, target_layer, per_pixel=False):
        fmaps = self._find(self.fmap_total, target_layer)
        grads = self._find(self.grad_total, target_layer)
        if per_pixel:
            weights = grads
        else:
            weights = self._compute_grad_weights(grads)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam_max = gcam.max(dim=1, keepdim=True)[0]
        gcam_max[torch.where(gcam_max == 0)] = 1. # prevent divide by 0
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam_max
        gcam = gcam.view(B, C, H, W)

        return gcam

