import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random

import torch.optim as optim
from torch.optim import lr_scheduler

from collections import OrderedDict
from tqdm import tqdm

from torchsummary import summary

from albumentations.augmentations import transforms, Normalize
from albumentations import Flip, RandomRotate90, Resize
from albumentations.core.composition import Compose, OneOf

from sklearn.model_selection import train_test_split

import argparse

import models
import metrics
import utils
from data import BUSIDataset

config = argparse.Namespace()

config.model = 'UNet'

config.SIZE = 256
config.batch_size = 8
config.num_workers = 2
config.n_channels = 1
config.lr = 0.0001
config.min_lr = 0.00001
config.epochs = 100
config.early_stopping = 25
config.patience = config.early_stopping
config.base_dir = ''
config.root_path = os.path.join(config.base_dir, 'breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/')
config.semantic = False
config.device = torch.device('cuda:0')
config.classes = ['normal', 'benign', 'malignant']
config.labels = []
config.num_classes = 3 if config.semantic else 1
config.loss = 'BCEDiceLoss'
config.optimizer = 'Adam'
config.scheduler = 'CosineAnnealingLR'
config.weight_decay = 1e-4
config.momentum = 0.9
config.nesterov = False

if config.semantic:
    config.labels = config.classes
else:
    config.labels = ['cancer']

def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': utils.AverageMeter(),
                  'iou': utils.AverageMeter(),
                   'dice': utils.AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        iou,dice = metrics.iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': utils.AverageMeter(),
                  'iou': utils.AverageMeter(),
                   'dice': utils.AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            
            output = model(input)
            loss = criterion(output, target)
            iou,dice = metrics.iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def showPredictions(count, X_val, y_val, model):
    for _ in range(count):
        idx = random.randint(0, len(X_val) - 1)

        image, mask = utils.load_tensor(X_val[idx], y_val[idx], size=config.SIZE)
        mask = mask.unsqueeze(0)
        image = image.unsqueeze(0)
        vanilla_prediction = model(image)
        utils.compare_prediction(image, mask, vanilla_prediction)
        print(f"Dice Score: {metrics.dice_coef(mask, vanilla_prediction)}")

def main():
    if config.model is not None and not os.path.exists('models/%s' % config.model):
        os.makedirs('models/%s' % config.model)

    print('-' * 20)
    for key, value in config.__dict__.items():
        print('%s: %s' % (key, value))
    print('-' * 20)


    criterion = metrics.__dict__[config.loss]().cuda()
    model = models.__dict__[config.model](n_channels=config.n_channels,
                                          n_classes=config.num_classes,
                                          n_layers=5).cuda()


    summary(model, (config.n_channels, config.SIZE, config.SIZE))

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(
            params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=config.lr, momentum=config.momentum,
                              nesterov=config.nesterov, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr)
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.factor, patience=config.patience,
                                                   verbose=1, min_lr=config.min_lr)
    else:
        scheduler = None

    image_paths, mask_paths = utils.load_dataset(config.root_path)
    print(f"Total Images: {len(image_paths)}, Masks: {len(mask_paths)}")
    
    train_transform = Compose([
        RandomRotate90(),
        Flip(),
    ])

    # image_paths, mask_paths = image_paths[:20], mask_paths[:20] # overfitting
    X_train, X_val, y_train, y_val = train_test_split(image_paths, mask_paths,  test_size=0.2, random_state=42)

    train_dataset = BUSIDataset(image_paths=X_train, mask_paths=y_train, size=config.SIZE, transform=train_transform)
    val_dataset = BUSIDataset(image_paths=X_val, mask_paths=y_val, size=config.SIZE, transform=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False)

    log = OrderedDict([
        ('train_lr', []),
        ('train_loss', []),
        ('train_iou', []),
        ('train_dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config.epochs):
        print('Epoch [%d/%d]' % (epoch, config.epochs))

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        if config.scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif config.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print(f"loss {train_log['loss']:.4f} - iou {train_log['iou']:.4f} - dice {train_log['dice']:.4f} \
              - val_loss {val_log['loss']:.4f} - val_iou {val_log['iou']:.4f} - val_dice {val_log['dice']:.4f}") 

        print(train_log, val_log)
        log['train_lr'].append(config.lr)
        log['train_loss'].append(train_log['loss'])
        log['train_iou'].append(train_log['iou'])
        log['train_dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config.model, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config.model)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config.early_stopping >= 0 and trigger >= config.early_stopping:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

    print(f"Val Loss: {log['val_loss'][-1]:.4f} - Val IOU: {log['val_iou'][-1]:.4f} - Val Dice: {log['val_dice'][-1]:.4f}")
    utils.plot_curves(config.model)
    # showPredictions(count=10, X_val=X_val, y_val=y_val, model=model)

if __name__ == '__main__':
    main()