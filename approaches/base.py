import os
import torch
from tqdm import tqdm
from torch import nn, optim
import wandb
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import gc

import utils.general_utils as gu
import utils.loss_utils as lu
import datasets


class Base():
    def __init__(self, config, dataloaders):
        self.CFG = config
        self.train_dataloader, self.val_dataloader = dataloaders[0], dataloaders[1]
        self.device      = 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'
        self.num_classes = self.CFG.DATA.NUM_CLASSES

        if self.num_classes == 2 and self.CFG.DATA.DATASET != 'coco_gender' and \
           not self.CFG.DATA.SEPARATE_CLASSES:
            self.classifier_classes = 1
        elif self.num_classes == 2 and self.CFG.DATA.SEPARATE_CLASSES:
            self.classifier_classes = 2
        else:
            self.classifier_classes = self.num_classes

        print('Number classifier outputs: {}'.format(self.classifier_classes))
        self.test_checkpoint    = None
        self.sched              = None

        self.need_eval_grad = self.CFG.EXP.LOSSES.GRADIENT_OUTSIDE.COMPUTE or self.CFG.EXP.LOSSES.GRADIENT_OUTSIDE.LOG  \
            or self.CFG.EXP.LOSSES.GRADIENT_INSIDE.COMPUTE or self.CFG.EXP.LOSSES.GRADIENT_INSIDE.LOG \
            or self.CFG.EXP.LOSSES.GRADCAM.COMPUTE or self.CFG.EXP.LOSSES.GRADCAM.LOG
        self.need_input_grad = self.need_eval_grad # same for now

        self.return_fmaps = config.EXP.LOSSES.GRADCAM.COMPUTE or \
            config.EXP.LOSSES.GRADCAM.LOG

        self.initialize_model()
        self.initialize_optimizers_and_schedulers()

        # Override the following defaults, if needed:
        self.group_names   = None
        self.label_mapping = None
        self.class_weights = None
        self.group_weights = None
        self.enforce_binary_train     = False # for COCO gender
        self.enforce_binary_eval      = False # for COCO gender
        self.no_penalty_person_eval   = False # for COCO gender
        if self.CFG.DATA.DATASET == 'waterbirds':
            from datasets.waterbirds import GROUP_NAMES
            self.group_names   = GROUP_NAMES
            self.label_mapping = datasets.waterbirds.get_label_mapping()
            if self.CFG.DATA.USE_CLASS_WEIGHTS:
                from datasets.waterbirds import get_loss_upweights
                self.class_weights = get_loss_upweights(
                    bias_fraction=self.CFG.DATA.CONFOUNDING_FACTOR,
                    mode='per_class'
                ).to(self.device)
            if self.CFG.DATA.USE_GROUP_WEIGHTS:
                from datasets.waterbirds import get_loss_upweights
                self.group_weights = get_loss_upweights(
                    bias_fraction=self.CFG.DATA.CONFOUNDING_FACTOR,
                    mode='per_group'
                ).to(self.device)
        elif self.CFG.DATA.DATASET == 'waterbirds_background':
            from datasets.waterbirds_background_task import GROUP_NAMES
            self.group_names   = GROUP_NAMES
            self.label_mapping = datasets.waterbirds_background_task.get_label_mapping()
            if self.CFG.DATA.USE_CLASS_WEIGHTS:
                from datasets.waterbirds_background_task import get_loss_upweights
                self.class_weights = get_loss_upweights(
                    bias_fraction=self.CFG.DATA.CONFOUNDING_FACTOR,
                    mode='per_class'
                ).to(self.device)
            if self.CFG.DATA.USE_GROUP_WEIGHTS:
                from datasets.waterbirds_background_task import get_loss_upweights
                self.group_weights = get_loss_upweights(
                    bias_fraction=self.CFG.DATA.CONFOUNDING_FACTOR,
                    mode='per_group'
                ).to(self.device)
        elif self.CFG.DATA.DATASET == 'coco_gender':
            if self.CFG.DATA.NUM_CLASSES == 2:
                assert self.CFG.DATA.BINARY_EVAL
            self.label_mapping = datasets.coco.get_label_mapping()
            if self.CFG.DATA.USE_CLASS_WEIGHTS:
                from datasets.coco import get_loss_upweights
                self.class_weights = get_loss_upweights(
                    self.CFG.DATA.MIN_NEEDED,
                    binary=self.CFG.DATA.BINARY_TRAIN,
                    only_woman=self.CFG.DATA.ONLY_UPWEIGHT_WOMAN
                ).to(self.device)
            else:
                self.class_weights = None
            self.enforce_binary_train = self.CFG.DATA.BINARY_TRAIN
            self.enforce_binary_eval  = self.CFG.DATA.BINARY_EVAL
            self.no_penalty_person_eval = self.CFG.DATA.NO_PENALTY_PERSON_PRED_EVAL
        elif self.CFG.DATA.DATASET == 'food_subset':
            self.label_mapping = np.array(sorted(self.CFG.DATA.CLASSES))
        else:
            raise NotImplementedError

        if self.label_mapping is not None:
            self.class_names = self.label_mapping
        else:
            self.class_names = np.array(
                ['label_{}'.format(int(label)) for label in range(self.num_classes)]
            )

        print('class weights: {}'.format(self.class_weights))
        print('group weights: {}'.format(self.group_weights))

        reduction = 'mean' if self.group_weights is None else 'none'
        if self.classifier_classes == 1:
            self.criterion = nn.BCELoss(reduction=reduction)
            self.activation = nn.Sigmoid()
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                reduction=reduction
            )
            self.activation = nn.Softmax(dim=1)

        self.loss_cfg = {
            'CLASSIFICATION': {
                'criterion': self.criterion,
                'activation': self.activation,
                'num_classes': self.classifier_classes,
                'group_weights': self.group_weights
            }
        }
        for loss_name, loss_settings in self.CFG.EXP.LOSSES.items():
            if loss_name in ['CLASSIFICATION', 'ABN_CLASSIFICATION']:
                    continue # ABN uses same settings as CLASSIFICATION
            if loss_settings['COMPUTE'] and loss_settings['LOG']:
                print('Error for {} loss: Both "COMPUTE" and "LOG" settings should not be True.'.format(
                    loss_name)
                )
                raise Exception
            if (loss_settings['COMPUTE'] or loss_settings['LOG']):
                if loss_name in ['GRADIENT_OUTSIDE', 'GRADIENT_INSIDE',
                                 'ABN_SUPERVISION', 'GRADCAM']:
                    if loss_settings['CRITERION'] == 'L1':
                        criterion = nn.L1Loss()
                    elif loss_settings['CRITERION'] == 'L2':
                        criterion = nn.MSELoss()
                    else:
                        raise NotImplementedError
                    self.loss_cfg[loss_name] = {
                        'criterion': criterion,
                    }
                    if loss_name in ['ABN_SUPERVISION', 'GRADCAM']:
                        self.loss_cfg[loss_name]['mode'] = loss_settings['MODE']
                else:
                    raise NotImplementedError


    # *** TO BE IMPLEMENTED BY CHILD CLASS ***
    def initialize_model(self):
        raise NotImplementedError

    def initialize_optimizers_and_schedulers(self):
        raise NotImplementedError

    def move_model_to_device(self):
        # Small hack for # of devices. Used to be torch.cuda.device_count(),
        # but we need to use only one GPU for test, and we can modify the environment variable,
        # but don't know how/if we can modify output of torch.cuda.device_count()
        if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
            print('>>> USING DATAPARALLEL')
            self.net = torch.nn.DataParallel(self.net)
        self.net.to(self.device)

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

    # *** OVERWRITE THE FOLLOWING IF NOT STANDARD ***
    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        state_dict = gu.check_module_state_dict(checkpoint['model_state_dict'])
        self.net.load_state_dict(state_dict)
        print('Loaded checkpoint {}'.format(checkpoint_file))

    def save_checkpoint(self, save_path, epoch, val_acc):
        print('Saving to {}'.format(save_path))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'val_acc': round(val_acc, 2)
            }, save_path
        )

    def forward(self, batch):
        inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
        if self.need_input_grad:
            inputs.requires_grad = True
        outputs = self.net(inputs)
        output_dict = {'logits': outputs, 'inputs': inputs, 'labels': labels}
        return output_dict


    def train_batch(self, batch, metrics):
        output_dict = self.forward(batch)
        inputs, labels = output_dict['inputs'], output_dict['labels']

        loss, metrics = self.calc_loss(
            metrics, 'train', batch, inputs, output_dict, labels
        )

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        _, _, preds = gu.calc_preds(
            output_dict['logits'],
            self.activation,
            self.classifier_classes,
            enforce_binary=self.enforce_binary_train
        )

        metrics['train_acc'].update(
            (preds == labels).sum().item() / inputs.shape[0], n=inputs.shape[0]
        )
        metrics = self.eval_by_class(metrics, labels, preds, mode='train')

        if self.group_names is not None:
            metrics = self.eval_groups(metrics, 'train', batch, preds, labels)

        del output_dict, preds
        return metrics

    def eval_batch(self, batch, metrics, mode='val'):
        assert mode in ['val', 'test']
        output_dict = self.forward(batch)
        inputs, labels = output_dict['inputs'], output_dict['labels']
        if mode == 'val':
            loss, metrics = self.calc_loss(metrics, 'val', batch, inputs, output_dict, labels)

        _, _, preds = gu.calc_preds(
            output_dict['logits'],
            self.activation,
            self.classifier_classes,
            enforce_binary=self.enforce_binary_eval
        )

        metrics['{}_acc'.format(mode)].update(
            (preds == labels).sum().item() / preds.shape[0], n=preds.shape[0]
        )
        metrics = self.eval_by_class(metrics, labels, preds, mode=mode)

        if self.group_names is not None:
            metrics = self.eval_groups(metrics, mode, batch, preds, labels)

        del output_dict, preds
        return metrics

    def get_metric_names(self):
        return [
            'train_acc',
            'val_acc',
            'train_cls_loss',
            'val_cls_loss',
            'train_total_loss',
            'val_total_loss',
            'balanced_train_acc',
            'balanced_val_acc'
        ]


    def setup_train(self):
        """
        Provide option for subclasses to perform any setup
        with test input arguments.
        """
        pass

    # *** STANDARD FUNCTIONS ***
    def train(self):
        self.setup_train()
        gu.init_wandb(self.CFG)

        start_epoch = 0

        # Logging setup
        metric_names = self.get_metric_names()
        if self.group_names is not None:
            for g in self.group_names:
                metric_names.append('{}_train_acc'.format(g))
                metric_names.append('{}_val_acc'.format(g))
        metrics = {name: gu.AverageMeter() for name in metric_names}

        wandb.run.summary['best_val_acc'] = -1

        for epoch in range(start_epoch, self.CFG.EXP.NUM_EPOCHS):
            print('EPOCH: {}\n'.format(epoch))
            for m in metrics.values():
                m.reset()

            print('>>> TRAINING \n')
            self.net.train()
            for i, data in enumerate(tqdm(self.train_dataloader)):
                metrics = self.train_batch(data, metrics)

            # ********** End of epoch val/logging **********
            if self.sched is not None:
                self.sched.step()

            print('>>> VALIDATING \n')
            metrics = self.validate(metrics)

            print('>>> LOGGING \n')

            # Calculate balanced accs
            _, metrics = self.compute_balanced_class_acc(metrics, mode='train')
            _, metrics = self.compute_balanced_class_acc(metrics, mode='val')

            # Check that balanced accs are computed correctly
            self.check_balanced_accs(metrics, 'train', balance_type='class')
            self.check_balanced_accs(metrics, 'val',   balance_type='class')

            for metric_name, metric_meter in metrics.items():
                print('{}: {}'.format(metric_name, metric_meter.avg))
                wandb.log({metric_name: metric_meter.avg}, step=epoch)

            print('TRAIN ACC:        {}'.format(metrics['train_acc'].avg))
            print('VAL ACC:          {}'.format(metrics['val_acc'].avg))
            print('BALANCED VAL ACC: {}'.format(metrics['balanced_val_acc'].avg))

            # Saving models, based on balanced val acc
            val_acc = metrics['balanced_val_acc'].avg
            if (self.CFG.LOGGING.SAVE_EVERY >= 1 and (epoch + 1) % self.CFG.LOGGING.SAVE_EVERY == 0) or \
               (self.CFG.LOGGING.SAVE_LAST and (epoch + 1) == self.CFG.EXP.NUM_EPOCHS):
                save_path = os.path.join(
                    wandb.run.dir,
                    'epoch-{}-{}valacc-{}.ckpt'.format(
                        epoch,
                        'balanced-' if self.group_names is not None else '',
                        round(val_acc, 2)
                    )
                )
                #self.test_checkpoint = save_path
                self.save_checkpoint(save_path, epoch, val_acc)

            if val_acc > wandb.run.summary['best_val_acc']:
                wandb.run.summary['best_val_acc'] = val_acc
                wandb.run.summary['best_val_epoch'] = epoch
                for cls in self.class_names:
                    wandb.run.summary['best_val_{}'.format(cls)] = metrics['val_acc_{}'.format(cls)].avg
            if self.CFG.LOGGING.SAVE_BEST and val_acc >= wandb.run.summary['best_val_acc']:
                # Delete file holding previous best model. Instead of having static filename for best model,
                # we save the val_acc in the filename for convenience in understanding the run.
                gu.del_prev_best_model_file(wandb.run.dir)
                save_path = os.path.join(wandb.run.dir,
                                         'best_{}valacc_{}_epoch_{}.ckpt'.format(
                                             'balanced_' if self.group_names is not None else '',
                                             round(val_acc, 2), epoch)
                )
                # Overwrites previous test_checkpoint field
                self.test_checkpoint = save_path
                self.save_checkpoint(save_path, epoch, val_acc)

        del self.opt, self.sched
        gc.collect()
        torch.cuda.empty_cache()

    def validate(self, metrics):
        self.net.eval()
        torch.set_grad_enabled(self.need_eval_grad)

        for i, data in enumerate(tqdm(self.val_dataloader)):
            metrics = self.eval_batch(data, metrics, mode='val')

        torch.set_grad_enabled(True)
        return metrics

    def setup_test(self, test_dataloader, checkpoint_file):
        """
        Provide option for subclasses to perform any setup
        with test input arguments.
        """
        pass

    def test(self, test_dataloader, checkpoint_file):
        """
        Load checkpoint and evaluate accuracy on test set.
        """
        self.setup_test(test_dataloader, checkpoint_file)
        if torch.cuda.device_count() > 1:
            # Can't use DataParallel w/ batch size of 1.
            # So reinitialize model on one GPU before loading checkpoint.
            if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
                print('>>> MOVING MODEL TO ONE GPU FOR TEST')
                devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
                self.initialize_model()

        self.load_checkpoint(checkpoint_file)

        metric_names = ['test_acc']
        if self.group_names is not None:
            metric_names.append('balanced_test_acc')
            for g in self.group_names:
                metric_names.append('{}_test_acc'.format(g))
        metrics = {name: gu.AverageMeter() for name in metric_names}

        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_dataloader)):
                metrics = self.eval_batch(
                    data,
                    metrics,
                    mode='test',
                )
        # Test: compute balanced performance by group if available.
        if self.group_names is not None:
            _, metrics = self.compute_balanced_group_acc(metrics, mode='test')
            self.check_balanced_accs(metrics, 'test', balance_type='group')
        else:
            _, metrics = self.compute_balanced_class_acc(metrics, mode='test')
            self.check_balanced_accs(metrics, 'test', balance_type='class')

        # COCO gender extra eval
        if self.CFG.EXP.APPROACH == 'coco_gender' or self.CFG.EXP.APPROACH == 'coco_abn' :
            metrics = self.compute_ratio_divergence(
                metrics,
                split='test',
                binary=self.CFG.DATA.BINARY_EVAL
            )

        print('TEST SET RESULTS FOR CHECKPOINT {}'.format(checkpoint_file))
        for metric_name, metric_meter in metrics.items():
            if metric_name != 'labels' and metric_name != 'preds':
                if type(metric_meter) == gu.AverageMeter:
                    print('{}: {}'.format(metric_name, metric_meter.avg))
                else:
                    print('{}: {}'.format(metric_name, metric_meter))

        if wandb.run is not None:
            for metric_name, metric_meter in metrics.items():
                if type(metric_meter) == gu.AverageMeter:
                    wandb.run.summary[metric_name] = metric_meter.avg
                else:
                    wandb.run.summary[metric_name] = metric_meter
        return metrics

    def eval_groups(self, metrics, split, batch, preds, labels):
        groups = batch['group'].to(self.device)
        for i, group_name in enumerate(self.group_names):
            group_samples = torch.where(groups == i)[0]
            if len(group_samples) > 0:
                group_preds = preds[group_samples]
                group_labels = labels[group_samples]
                metrics['{}_{}_acc'.format(group_name, split)].update(
                    (group_preds==group_labels).sum().item() / len(group_samples),
                    n=len(group_samples)
                )
        return metrics

    def eval_by_class(self, metrics, labels, preds, mode='val'):
        unique_labels = torch.unique(labels)
        names = self.label_mapping if self.label_mapping is not None else unique_labels
        for label in unique_labels:
            label = int(label)
            if '{}_acc_{}'.format(mode, names[label]) not in metrics:
                metrics['{}_acc_{}'.format(mode, names[label])] = gu.AverageMeter()
            label_inds    = torch.where(labels == label)[0]
            label_preds   = preds[label_inds]
            label_samples = labels[label_inds]
            metrics['{}_acc_{}'.format(mode, names[label])].update(
                (label_preds==label_samples).sum().item() / len(label_samples),
                n=len(label_samples)
            )
        return metrics

    def compute_balanced_class_acc(self, metrics, mode='val'):
        names = self.class_names
        class_avgs = []
        for label in range(self.num_classes):
            key = '{}_acc_{}'.format(mode, names[label])
            if key in metrics:
                if metrics[key].count > 0:
                    avg = metrics[key].avg
                    class_avgs.append(avg)
        balanced_acc = np.array(class_avgs).mean()
        metrics['balanced_{}_acc'.format(mode)] = gu.AverageMeter()
        metrics['balanced_{}_acc'.format(mode)].update(balanced_acc)
        return balanced_acc, metrics


    def compute_balanced_group_acc(self, metrics, mode='val'):
        assert self.group_names is not None
        group_avgs = []
        for g in self.group_names:
            key = '{}_{}_acc'.format(g, mode)
            if key in metrics:
                if metrics[key].count > 0:
                    avg = metrics[key].avg
                    group_avgs.append(avg)
        balanced_acc = np.array(group_avgs).mean()
        metrics['balanced_{}_acc'.format(mode)] = gu.AverageMeter()
        metrics['balanced_{}_acc'.format(mode)].update(balanced_acc)
        return balanced_acc, metrics


    def check_balanced_accs(self, metrics, split, balance_type='class'):
        assert balance_type in ['class', 'group']
        if balance_type == 'group':
            # Balanced acc = average across groups
            accs = []
            for g in self.group_names:
                key = '{}_{}_acc'.format(g, split)
                if key in metrics:
                    if metrics[key].count > 0:
                        accs.append(metrics[key].avg)
        else:
            # Balanced acc = average across classes
            if self.label_mapping is not None:
                names = self.label_mapping
            else:
                names = np.array(
                    ['label_{}'.format(int(label)) for label in range(self.num_classes)]
                )
            accs = []
            for c in names:
                key = '{}_acc_{}'.format(split, c)
                if key in metrics:
                    if metrics[key].count > 0:
                        accs.append(metrics[key].avg)

        assert np.array(accs).mean() == metrics['balanced_{}_acc'.format(split)].avg

        print('BALANCED ACCS CHECKED FOR {}'.format(split.upper()))
        return

