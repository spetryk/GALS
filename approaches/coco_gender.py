"""
Evaluation code credit to:

Lisa Anne Hendricks, Kaylee Burns, Kate Saenko, Trevor Darrell, Anna Rohrbach.
Women Also Snowboard: Overcoming Bias in Captioning Models
ECCV 2018

https://github.com/kayburns/women-snowboard/tree/master/research/im2txt
"""

import os
import torch
from tqdm import tqdm
from torch import nn, optim
import wandb
import numpy as np
import matplotlib.pyplot as plt

import utils.general_utils as gu
import utils.loss_utils as lu
from approaches.generic_cnn import GenericCNN
from models.resnet import resnet18, resnet50

class COCOGenderCNN(GenericCNN):
    def __init__(self, config, dataloaders):
        super(COCOGenderCNN, self).__init__(config, dataloaders)

    def eval_by_class(self, metrics, labels, preds, mode='val'):
        unique_labels = torch.unique(labels)
        if self.label_mapping is not None:
            names = self.label_mapping
        else:
            names = np.array(['label_{}'.format(int(label)) for label in unique_labels])

        names = self.label_mapping if self.label_mapping is not None else unique_labels

        f_tp = 0.
        f_fp = 0.
        f_tn = 0.
        f_other = 0.
        f_total = 0.
        f_person = 0.

        m_tp = 0.
        m_fp = 0.
        m_tn = 0.
        m_other = 0.
        m_total = 0.

        for i,pred in enumerate(preds):
            label  = int(labels[i])
            pred   = int(pred)
            male   = label == 0
            female = label == 1
            pred_male = pred == 0
            pred_female   = pred == 1
            pred_other  = pred == 2

            if (female & pred_female):
                f_tp += 1
            if (male & pred_male):
                m_tp += 1
            if (male & pred_female):
                f_fp += 1
            if (female & pred_male):
                m_fp += 1
            if ((not female) & (not pred_female)):
                f_tn += 1
            if ((not male) & (not pred_male)):
                m_tn += 1
            if (female & pred_other):
                f_other += 1
            if (male & pred_other):
                m_other += 1
            if female:
                f_total += 1
            if male:
                m_total += 1

        metrics = self.add_metric_keys(metrics, mode)
        if m_total > 0:
            metrics['{}_male_correct'.format(mode)].update(    m_tp/m_total, n=m_total)
            metrics['{}_male_incorrect'.format(mode)].update(  f_fp/m_total, n=m_total)
            metrics['{}_male_other'.format(mode)].update(   m_other/m_total, n=m_total)
        if f_total > 0:
            metrics['{}_female_correct'.format(mode)].update(  f_tp/f_total, n=f_total)
            metrics['{}_female_incorrect'.format(mode)].update(m_fp/f_total, n=f_total)
            metrics['{}_female_other'.format(mode)].update( f_other/f_total, n=f_total)
            if (m_total + f_total) > 0:
                metrics['{}_all_correct'.format(mode)].update((m_tp+f_tp)/(m_total+f_total),
                                                              n=(m_total+f_total))
                metrics['{}_all_incorrect'.format(mode)].update((m_fp+f_fp)/(m_total+f_total),
                                                                n=(m_total+f_total))
                metrics['{}_all_other'.format(mode)].update((m_other+f_other)/(f_total+m_total),
                                                            n=(m_total+f_total))
        metrics['{}_ratio_numerator'.format(mode)].update(f_tp+f_fp)
        metrics['{}_ratio_denominator'.format(mode)].update(m_tp+m_fp)

        for label in unique_labels:
            label = int(label)
            if '{}_acc_{}'.format(mode, names[label]) not in metrics:
                metrics['{}_acc_{}'.format(mode, names[label])] = gu.AverageMeter()
            label_inds    = torch.where(labels == label)[0]
            label_preds   = preds[label_inds]
            label_samples = labels[label_inds]
            metrics['{}_acc_{}'.format(mode, names[label])].update(
                (label_preds==label_samples).sum().item() / len(label_samples),
                n=len(label_samples))
        return metrics

    def add_metric_keys(self, metrics, mode):
        keys = [
            '{}_male_correct'.format(mode),
            '{}_male_incorrect'.format(mode),
            '{}_male_other'.format(mode),
            '{}_female_correct'.format(mode),
            '{}_female_incorrect'.format(mode),
            '{}_female_other'.format(mode),
            '{}_all_correct'.format(mode),
            '{}_all_incorrect'.format(mode),
            '{}_all_other'.format(mode),
            '{}_ratio_numerator'.format(mode),
            '{}_ratio_denominator'.format(mode)
        ]
        for k in keys:
            if k not in metrics:
                metrics[k] = gu.AverageMeter()
        return metrics

    def compute_ratio_divergence(self, metrics, split='test', binary=False):
        """
        binary: True if network only predicted man and woman.
                False if predicted man, woman, and person.
        """
        # Ratio of num woman predictions / num man predictions
        assert '{}_ratio_numerator'.format(split) in metrics
        assert '{}_ratio_denominator'.format(split) in metrics
        ratio = metrics['{}_ratio_numerator'.format(split)].sum /   \
            metrics['{}_ratio_denominator'.format(split)].sum
        print('{} ratio: {}'.format(split, ratio))
        metrics['{}_ratio'.format(split)] = ratio

        # Divergence between woman and man (and person, if 3 classes)
        # prediction distribution
        woman_probs = [
            metrics['{}_female_correct'.format(split)].avg,
            metrics['{}_female_incorrect'.format(split)].avg
        ]

        if not binary:
            woman_probs.append(metrics['{}_female_other'.format(split)].avg)
        woman_probs = np.array(woman_probs)

        man_probs = [
            metrics['{}_male_correct'.format(split)].avg,
            metrics['{}_male_incorrect'.format(split)].avg
        ]
        if not binary:
            man_probs.append(metrics['{}_male_other'.format(split)].avg)
        man_probs = np.array(man_probs)

        m = 0.5*woman_probs + 0.5*man_probs

        outcome_divergence = 0.5*kl(woman_probs, m) + 0.5*kl(man_probs, m)

        print('{} outcome divergence: {}'.format(split, outcome_divergence))
        metrics['{}_outcome_divergence'.format(split)] = outcome_divergence
        return metrics

def kl(q, p):
    return np.sum(q*(np.log(q/p) / np.log(2)))

