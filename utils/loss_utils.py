import torch
import numpy as np
from torchvision import transforms

from utils import general_utils as gu
from utils import attention_utils as au

def calc_loss(metrics, split, batch, inputs, output_dict, labels, cfg, loss_cfg, device):
    loss = 0

    # Always compute classification loss
    if cfg.DATA.USE_GROUP_WEIGHTS:
        cls_loss = calc_classification_loss(loss_cfg['CLASSIFICATION']['criterion'],
                                            loss_cfg['CLASSIFICATION']['activation'],
                                            loss_cfg['CLASSIFICATION']['num_classes'],
                                            output_dict['logits'],
                                            labels,
                                            group_weights=loss_cfg['CLASSIFICATION']['group_weights'],
                                            group_labels=batch['group'],
                                            device=device)
    else:
        cls_loss = calc_classification_loss(loss_cfg['CLASSIFICATION']['criterion'],
                                            loss_cfg['CLASSIFICATION']['activation'],
                                            loss_cfg['CLASSIFICATION']['num_classes'],
                                            output_dict['logits'],
                                            labels)
    loss += cls_loss
    metrics['{}_cls_loss'.format(split)].update(cls_loss.item(), n=inputs.shape[0])

    # Precompute dy/dx if needed, so can be reused by various losses
    compute_grad = cfg.EXP.LOSSES.GRADIENT_OUTSIDE.COMPUTE or cfg.EXP.LOSSES.GRADIENT_OUTSIDE.LOG  \
        or cfg.EXP.LOSSES.GRADIENT_INSIDE.COMPUTE or cfg.EXP.LOSSES.GRADIENT_INSIDE.LOG
    if compute_grad:
        dy_dx = torch.autograd.grad(cls_loss, inputs, retain_graph=True, create_graph=True)
        assert len(dy_dx) == 1
        assert dy_dx[0].shape == inputs.shape
        dy_dx = dy_dx[0]
    else:
        dy_dx = None

    for loss_name, loss_settings in cfg.EXP.LOSSES.items():
        if loss_name == 'CLASSIFICATION':
            continue
        elif loss_settings['COMPUTE'] or loss_settings['LOG']:

            if loss_name == 'GRADIENT_OUTSIDE':
                gt_attentions, valid_dy_dx = get_gt_pred_attentions(
                    loss_settings, loss_name, batch, inputs, device,
                    pred_attentions=dy_dx)
                if gt_attentions is None:
                    continue
                grad_loss = input_gradient_loss(
                    loss_cfg['GRADIENT_OUTSIDE']['criterion'],
                    inputs, gt_attentions,
                    dy_dx=valid_dy_dx,
                    mode='suppress_outside'
                )
                grad_loss *= loss_settings['WEIGHT']
                if loss_settings['COMPUTE']:
                    loss += grad_loss

                # Logging
                key = '{}_dydx_outside_{}_loss'.format(split, loss_settings['GT'])
                if key not in metrics:
                    metrics[key] = gu.AverageMeter()
                metrics[key].update(grad_loss.item(), n=inputs.shape[0])

            elif loss_name == 'GRADIENT_INSIDE':
                gt_attentions, valid_dy_dx = get_gt_pred_attentions(
                    loss_settings, loss_name, batch, inputs, device,
                    pred_attentions=dy_dx
                )
                if gt_attentions is None:
                    continue
                grad_loss = input_gradient_loss(
                    loss_cfg['GRADIENT_INSIDE']['criterion'],
                    inputs, gt_attentions,
                    dy_dx=valid_dy_dx,
                    mode='match_inside'
                )
                grad_loss *= loss_settings['WEIGHT']
                if loss_settings['COMPUTE']:
                    loss += grad_loss

                # Logging
                key = '{}_dydx_inside_{}_loss'.format(split, loss_settings['GT'])
                if key not in metrics:
                    metrics[key] = gu.AverageMeter()
                metrics[key].update(grad_loss.item(), n=inputs.shape[0])

            elif loss_name == 'GRADCAM':
                fmaps  = output_dict['fmaps']
                logits = output_dict['logits']

                gcams  = au.compute_gradcam(fmaps, logits, labels, device)
                gt_attentions, valid_gcams = get_gt_pred_attentions(
                    loss_settings, loss_name, batch, inputs, device,
                    pred_attentions=gcams
                )
                if gt_attentions is None:
                    continue
                grad_loss = attention_map_loss(
                    loss_cfg['GRADCAM']['criterion'],
                    valid_gcams, gt_attentions,
                    mode=loss_cfg['GRADCAM']['mode']
                )
                grad_loss *= loss_settings['WEIGHT']

                if loss_settings['COMPUTE']:
                    loss += grad_loss

                # Logging
                key = '{}_gradcam_{}_loss'.format(split, loss_settings['GT'])
                if key not in metrics:
                    metrics[key] = gu.AverageMeter()
                metrics[key].update(grad_loss.item(), n=inputs.shape[0])

            elif loss_name == 'ABN_SUPERVISION':
                pred_attentions = output_dict['attention']
                gt_attentions, valid_pred_attentions = get_gt_pred_attentions(
                    loss_settings, loss_name, batch, inputs, device,
                    pred_attentions=pred_attentions
                )
                if gt_attentions is None:
                    continue

                att_loss = attention_map_loss(
                    loss_cfg['ABN_SUPERVISION']['criterion'],
                    valid_pred_attentions, gt_attentions,
                    mode=loss_cfg['ABN_SUPERVISION']['mode']
                )
                att_loss *= loss_settings['WEIGHT']
                if loss_settings['COMPUTE']:
                    loss += att_loss

                # Logging
                key = '{}_abn_att_{}_loss'.format(split, loss_settings['GT'])
                if key not in metrics:
                    metrics[key] = gu.AverageMeter()
                metrics[key].update(att_loss.item(), n=inputs.shape[0])


            elif loss_name == 'ABN_CLASSIFICATION':
                # Same criterion, activation, num_classes, & labels as normal classification loss.
                # Only difference is source of logits - they are output from the attention branch
                abn_cls_loss = calc_classification_loss(
                    loss_cfg['CLASSIFICATION']['criterion'],
                    loss_cfg['CLASSIFICATION']['activation'],
                    loss_cfg['CLASSIFICATION']['num_classes'],
                    output_dict['att_logits'],
                    labels
                )
                abn_cls_loss *= loss_settings['WEIGHT']
                if loss_settings['COMPUTE']:
                    loss += abn_cls_loss

                # Logging
                key = '{}_abn_cls_loss'.format(split)
                if key not in metrics:
                    metrics[key] = gu.AverageMeter()
                metrics[key].update(abn_cls_loss.item(), n=inputs.shape[0])

            else:
                raise NotImplementedError

    metrics['{}_total_loss'.format(split)].update(loss.item(), n=inputs.shape[0])

    return loss, metrics


def calc_classification_loss(
        criterion,
        activation,
        num_classifier_classes,
        logits,
        labels,
        group_weights=None,
        group_labels=None,
        device=None
):
    if num_classifier_classes == 1:
        preds = activation(logits)[:,0]
        gt = labels.float()
    else:
        preds = logits
        gt = labels.long()

    if group_weights is not None:
        assert group_labels is not None
        assert device is not None
        loss = group_weighted_cross_entropy(
            criterion,
            preds,
            gt,
            group_labels,
            group_weights,
            device
        )
    else:
        loss = criterion(preds, gt)

    return loss

def group_weighted_cross_entropy(criterion, inputs, labels, group_labels, group_weights, device):
    """
    inputs and labels are directly given to criterion, so should be correct type.
    criterion must have reduction as none.

    Returns scalar loss.
    """
    assert criterion.reduction == 'none'
    outputs        = criterion(inputs, labels)
    sample_weights = torch.ones(outputs.shape).to(device)
    group_labels   = group_labels.to(device)
    for group in torch.unique(group_labels):
        group_inds = torch.where(group_labels == group)[0].long()
        sample_weights[group_inds] = group_weights[group.long()]
    outputs *= sample_weights
    return outputs.mean()


def input_gradient_loss(criterion, inputs, gt_attentions, dy_dx=None, mode='suppress_outside'):
    """
    inputs: B x C x H x W input tensor
    y: scalar, usually classification loss, that dy/d_inputs is taken with.

    gt_attentions: same shape as input tensor.
      - Specifies where to KEEP gradients.
      - Should be already normalized b/w [0,1].
      - Where it is 1, dy/dx is not penalized.
      - Where it is b/w [0,1), dy/dx is suppressed by that factor.
      - If all 0's, equivalent to double backprop proposed by Drucker & LeCun [1992]
        (regularize input gradients everywhere)

    dy_dx: required arg.
      Gradient of desired output w.r.t. inputs.

    Options for mode:
      - "suppress_outside": penalize non-zero dy/dx where gt_attentions is not 1
      - "match_inside": encourage dy/dx to match gt_attentions where gt_attentions is not 0
    """
    assert inputs.requires_grad
    assert dy_dx is not None

    if mode == 'suppress_outside':
        loss   = criterion(dy_dx, dy_dx*gt_attentions)
    elif mode == 'match_inside':
        target = ((1-gt_attentions)*dy_dx) + gt_attentions
        loss   = criterion(dy_dx, target)
        print(loss)
    else:
        raise NotImplementedError

    return loss

def attention_map_loss(criterion, pred_attentions, gt_attentions, mode='match'):
    """
    pred_attentions: B x C x feature_map_height x feature_map_width
    gt_attentions: B x C x input_image_height x input_image_width

    gt_attentions is resized down to pred_attentions hxw.

    Different modes:
      - match: attention should match GT.
      - suppress_outside: only attention outside the segmentation is penalized.
    """

    if gt_attentions.shape[-2:] != pred_attentions.shape[-2:]:
        gt_attentions = transforms.functional.resize(gt_attentions, pred_attentions.shape[-2:])
    assert gt_attentions.shape == pred_attentions.shape

    if mode == 'match':
        loss = criterion(pred_attentions, gt_attentions)
    elif mode == 'suppress_outside':
        loss = criterion(pred_attentions, pred_attentions*gt_attentions)
    else:
        raise NotImplementedError

    return loss

def get_valid_attentions(gt_attentions, pred_attentions, combine_att_mode, loss_name=None):
    gt_attentions, valid_inds = au.parse_attention(
        gt_attentions,
        shape=pred_attentions.shape[-2:],
        mode=combine_att_mode
    )
    if len(valid_inds) == 0:
        # No valid GT attentions in batch. Don't calculate a loss.
        print('no valid GT in batch {} loss'.format(loss_name))
        return None, None
    else:
        valid_preds = pred_attentions[valid_inds == 1]
        return gt_attentions, valid_preds


def get_gt_pred_attentions(loss_settings, loss_name, batch, inputs, device,
                           pred_attentions=None, invert_attention=False):
    valid_pred_attentions = pred_attentions
    if loss_settings['GT'] == 'zeros':
        gt_attentions = torch.zeros(inputs.shape)
    elif loss_settings['GT'] == 'segmentation':
        gt_attentions = batch['seg'].to(device)
        gt_attentions = gt_attentions.bool().float()
    elif loss_settings['GT'] == 'bbox':
        gt_attentions = batch['bbox'].to(device)
        gt_attentions = gt_attentions.bool().float()
    else:
        gt_attentions = batch['attention'].to(device)
        gt_attentions, valid_pred_attentions = get_valid_attentions(
            gt_attentions,
            valid_pred_attentions,
            loss_settings['COMBINE_ATT_MODE'],
            loss_name=loss_name
        )
    if 'INVERT' in loss_settings and loss_settings['INVERT'] and gt_attentions is not None:
        gt_attentions = 1 - gt_attentions
    return gt_attentions, valid_pred_attentions
