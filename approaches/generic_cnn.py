import os
import torch
from tqdm import tqdm
from torch import nn, optim
import wandb
import numpy as np
import matplotlib.pyplot as plt

import utils.general_utils as gu
import utils.loss_utils as lu
from approaches.base import Base
from models.resnet import resnet18, resnet50

class GenericCNN(Base):
    def __init__(self, config, dataloaders):
        super(GenericCNN, self).__init__(config, dataloaders)

    def forward(self, batch):
        inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
        if self.need_input_grad:
            inputs.requires_grad = True
        if self.return_fmaps:
            outputs, fmaps = self.net(inputs)
            output_dict = {'logits': outputs, 'fmaps': fmaps}
        else:
            outputs = self.net(inputs)
            output_dict = {'logits': outputs}
        output_dict['inputs'] = inputs
        output_dict['labels'] = labels
        return output_dict

    def initialize_model(self):
        pretrain = self.CFG.EXP.PRETRAINED
        print('Using pretrained model: {}'.format(pretrain))
        if self.CFG.EXP.MODEL == 'resnet18':
            self.net = resnet18(pretrained=pretrain, return_fmaps=self.return_fmaps)
        elif self.CFG.EXP.MODEL == 'resnet50':
            self.net = resnet50(pretrained=pretrain, return_fmaps=self.return_fmaps)
        else:
            raise NotImplementedError

        if hasattr(self.net, 'fc'):
            # Replace last FC layer with FC w/ custom num classes
            self.net.fc = nn.Linear(
                in_features=self.net.fc.in_features,
                out_features=self.classifier_classes, bias=True
            )
        self.move_model_to_device()

    def initialize_optimizers_and_schedulers(self):
        base_params = []
        fc_params = []
        for name, param in self.net.named_parameters():
            if 'fc' in name:
                fc_params.append(param)
            else:
                base_params.append(param)
        param_list = [
            {'params': base_params, 'lr': self.CFG.EXP.BASE.LR},
            {'params': fc_params,  'lr': self.CFG.EXP.CLASSIFIER.LR},
        ]

        if self.CFG.EXP.OPTIMIZER == 'SGD':
            print('USING SGD OPTIMIZER')
            self.opt = optim.SGD(
                param_list,
                momentum=self.CFG.EXP.MOMENTUM,
                weight_decay=self.CFG.EXP.WEIGHT_DECAY
            )
        elif self.CFG.EXP.OPTIMIZER == 'ADAMW':
            print('USING ADAMW OPTIMIZER')
            self.opt = optim.AdamW(
                param_list,
                weight_decay=self.CFG.EXP.WEIGHT_DECAY,
                amsgrad=False
            )
        else:
            raise NotImplementedError




