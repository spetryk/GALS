import os
import torch
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
import wandb

import numpy as np
import matplotlib.pyplot as plt

import utils.general_utils as gu
import utils.loss_utils as lu
#from approaches.base import Base
#from approaches.generic_cnn import GenericCNN
from approaches.coco_gender import COCOGenderCNN
from models.resnet_abn import resnet50 as resnet50_abn

class COCOABN(COCOGenderCNN):
    def __init__(self, config, dataloaders):
        super(COCOABN, self).__init__(config, dataloaders)

        if self.CFG.EXP.PROVIDED_ATT != "NONE" and \
           (self.CFG.EXP.LOSSES.ABN_SUPERVISION.COMPUTE or self.CFG.EXP.LOSSES.ABN_SUPERVISION.LOG \
           or self.CFG.EXP.LOSSES.ABN_CLASSIFICATION.COMPUTE or self.CFG.EXP.LOSSES.ABN_CLASSIFICATION.LOG ):
            print('ERROR: Cannot provide ABN attention as well as log/compute the \
            attention classification or supervision loss')
            raise Exception

        self.calc_abn_cls_loss = self.CFG.EXP.LOSSES.ABN_CLASSIFICATION.COMPUTE or \
            self.CFG.EXP.LOSSES.ABN_CLASSIFICATION.LOG

        if self.CFG.EXP.PROVIDED_ATT != "NONE":
            raise NotImplementedError

    def initialize_model(self):
        pretrain = self.CFG.EXP.PRETRAINED
        print('Using pretrained model: {}'.format(pretrain))
        if self.CFG.EXP.MODEL == 'resnet50_abn':
            self.net = resnet50_abn(
                pretrained=False,
                num_classes=self.classifier_classes,
                add_after_attention=True
            )
            state_dict = torch.load('weights/resnet50_abn_imagenet.pth.tar')['state_dict']
            state_dict = gu.check_module_state_dict(state_dict, force_remove_module=True)
            if self.classifier_classes != 1000:
                # Remove imagenet-specific weights from state dict
                # present in the attention branch.
                new_dict = {}
                for k,v in state_dict.items():
                    if 'att_conv' in k or 'bn_att2' in k or 'fc' in k:
                        print('Removing {} from pretained weights, with weight shape {}'.format(k, v.shape))
                        continue
                    new_dict[k] = v
                state_dict = new_dict
            missing_keys, unexpected_keys = self.net.load_state_dict(state_dict, strict=False)
            print('Keys missing in state dict:    {}'.format(missing_keys))
            print('Unexpected keys in state dict: {}'.format(unexpected_keys))
        else:
            raise NotImplementedError

        self.move_model_to_device()

    def get_metric_names(self):
        return [
            'train_acc',
            'val_acc',
            'train_abn_acc',
            'val_abn_acc',
            'train_cls_loss',
            'val_cls_loss',
            'train_total_loss',
            'val_total_loss',
            'test_abn_acc',
            'balanced_train_acc',
            'balanced_val_acc'
        ]

    def calc_loss(self, metrics, split, batch, inputs, output_dict, labels):
        loss, metrics = lu.calc_loss(
            metrics=metrics,
            split=split,
            batch=batch,
            inputs=inputs,
            output_dict=output_dict,
            labels=labels,
            cfg=self.CFG,
            loss_cfg=self.loss_cfg,
            device=self.device
        )
        return loss, metrics


    def forward(self, batch):
        inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
        provided_att = self.get_provided_att(batch)
        if self.CFG.EXP.MODEL == 'resnet50_abn':
            att_outputs, outputs, [att, _, _] = self.net(
                inputs,
                provided_att=provided_att,
            )
        else:
            outputs = self.net(
                inputs,
                provided_att=provided_att,
            )
            att_outputs = None
            att = None
        output_dict = {
            'logits': outputs,
            'att_logits': att_outputs,
            'attention': att,
            'inputs': inputs,
            'labels': labels
        }
        return output_dict


    def get_provided_att(self, batch):
        if self.CFG.EXP.PROVIDED_ATT != "NONE":
            if self.CFG.EXP.PROVIDED_ATT == 'segmentation':
                seg = batch['seg'].to(self.device)
                seg = seg.bool().float()
                seg = transforms.functional.resize(
                    seg, list(self.CFG.EXP.PROVIDED_ATT_SHAPE)
                )
                provided_att = seg
            else:
                raise NotImplementedError
        else:
            provided_att = None
        return provided_att
