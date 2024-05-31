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
import utils, metrics

from torchsummary import summary

from albumentations.augmentations import transforms
from albumentations import Flip, RandomRotate90, Resize
from albumentations.core.composition import Compose, OneOf

def load_image(path, size):
    image = cv2.imread(path)
    image = cv2.resize(image, (size, size))
    image = image[:, :, 0]
    image = image / 255.0
    image = np.expand_dims(image, 0)
    image = image.astype(np.double)
    return image

def load_mask(paths, size, classes = ['normal', 'benign', 'malignant'], semantic=False, num_classes=3):
    mask = np.zeros((size, size), dtype=int)

    for path in paths:
        for i in range(3):
            if classes[i] in path:
                cur_mask = cv2.imread(path)
                cur_mask = np.mean(cur_mask, axis=-1)
                cur_mask = cv2.resize(cur_mask, (size, size))
                mask[cur_mask != 0] = i if semantic else 1

    if semantic:
        mask = np.eye(num_classes)[mask]
        mask = mask.astype(np.double)
        mask = np.transpose(mask, (2, 0, 1))
    else:
        mask = np.expand_dims(mask, 0)
    return mask

def load_tensor(image_path, mask_path, device='cuda', size=256):
    image, mask = load_image(image_path, size=size), load_mask(mask_path, size=size)
    image = torch.tensor(image).to(device=device, dtype=torch.float)
    mask = torch.tensor(mask).to(device, dtype=torch.float)
    return image, mask

def show_image(img, msk, labels=['cancer'], semantic=False, threshold=False):
    fig, axs = plt.subplots(1, 1 + len(labels), figsize=(8, 3))
    img = np.squeeze(img)

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("image")

    for i in range(len(labels)):
        if semantic and threshold:
            # Create a binary mask based on the channel with the highest value
            binary_mask = np.argmax(msk, axis=0) == i
            axs[i + 1].imshow(binary_mask, vmin=0, vmax=1, cmap='gray')
        else:
            axs[i + 1].imshow(msk[i], vmin=0, vmax=1, cmap='gray')
        axs[i + 1].set_title(f"{labels[i]}")

    plt.show()

def show_mask(img, msk, cmap='gray', alpha=0.6): # only for binary segmentation
    img = np.squeeze(img)
    msk = np.squeeze(msk)

    msk = (msk >= 0.5)

    plt.imshow(img, cmap='gray')
    plt.imshow(msk, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.show()

# comparison

def compare_prediction(img, mask, pred, pred2 = None, classes=['normal', 'benign', 'malignant'], semantic=False):
    img = img.cpu().numpy()
    mask = mask.cpu().numpy()
    pred = pred.detach().cpu().numpy()

    img = np.squeeze(img)
    mask = np.squeeze(mask, axis=0)
    pred = np.squeeze(pred, axis=0)

    fig, axs = plt.subplots(1, 3 + (pred2 != None), figsize=(12, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Image")

    if semantic:
        mask = mask[1:, :, :]
        gt_label = mask.sum(axis=(1, 2)).argmax()  # 1-based indexing
        axs[1].imshow(mask[gt_label, :, :], cmap='gray', vmin=0, vmax=1)
        axs[1].set_title(f"Ground Truth: {classes[gt_label + 1]}")

        pred = pred[1:, :, :, ]
        pred_label = pred.sum(axis=(1, 2)).argmax()  # 1-based indexing
        axs[2].imshow(pred[pred_label, :, :], cmap='gray', vmin=0, vmax=1)
        axs[2].set_title(f"Prediction: {classes[pred_label + 1]}")

        if pred2 != None:
            pred2 = pred2.detach().cpu().numpy()
            pred2 = np.squeeze(pred2, axis=0)
            pred2 = pred2[1:, :, :, ]
            pred2_label = pred2.sum(axis=(1, 2)).argmax()
            axs[3].imshow(pred2[pred2_label, :, :], cmap='gray', vmin=0, vmax=1)
            axs[3].set_title(f"Prediction 2: {classes[pred2_label + 1]}")

    else:
        pred = np.where(pred >= 0.5, 1, 0)

        axs[1].imshow(mask[0], cmap='gray', vmin=0, vmax=1)
        axs[1].set_title(f"Ground Truth")

        axs[2].imshow(pred[0], cmap='gray', vmin=0, vmax=1)
        axs[2].set_title(f"Prediction")

        if pred2 != None:
            pred2 = pred2.detach().cpu().numpy()
            pred2 = np.squeeze(pred2, axis=0)
            pred2 = np.where(pred2 >= 0.5, 1, 0)
            axs[3].imshow(pred2[0], cmap='gray', vmin=0, vmax=1)
            axs[3].set_title(f"Prediction 2")


def show_prediction(img, pred, semantic = False, num_classes = 3, labels = ['cancer'], threshold = False):
    img = img.cpu().numpy()
    pred = pred.detach().cpu().numpy()

    img = np.squeeze(img)
    pred = np.squeeze(pred, axis=0)


    if semantic and threshold:
        argmax_result = np.argmax(pred, axis=0)
        pred = np.eye(num_classes)[argmax_result]
        pred = np.transpose(pred, axes=(2, 0, 1))

    fig, axs = plt.subplots(1, 1 + len(labels), figsize=(12, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Image")

    for i in range(len(labels)):
        axs[i+1].imshow(pred[i], cmap='gray')
        axs[i+1].set_title(f"Label: {labels[i]}")

def load_dataset(root_path):
    all_images = glob.glob(os.path.join(root_path, "**/*.png"), recursive=True)

    image_paths = []
    mask_paths = []

    for path in all_images:
        if 'normal' in path or 'mask' in path:
            continue

        image_paths.append(path)
        cur_mask_path = []
        mask_path_1 = path.replace('.png', '_mask.png')
        mask_path_2 = path.replace('.png', '_mask_1.png')
        cur_mask_path.append(mask_path_1)
        if os.path.exists(mask_path_2):
            cur_mask_path.append(mask_path_2)
        mask_paths.append(cur_mask_path)

    image_paths = np.array(image_paths)
    mask_paths = np.array(mask_paths, dtype=object)

    return image_paths, mask_paths

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_curves(model_name):
    log_data = pd.read_csv('models/%s/log.csv' % model_name)

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(log_data['train_loss'], label='Train Loss')
    plt.plot(log_data['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    # Plot training and validation IoU
    plt.subplot(1, 3, 2)
    plt.plot(log_data['train_iou'], label='Train IoU')
    plt.plot(log_data['val_iou'], label='Val IoU')
    plt.title('IoU')
    plt.legend()

    # Plot training and validation Dice score
    plt.subplot(1, 3, 3)
    plt.plot(log_data['train_dice'], label='Train Dice')
    plt.plot(log_data['val_dice'], label='Val Dice')
    plt.title('Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def show_predictions(count, X_val, y_val, model, size):
    for _ in range(count):
        idx = random.randint(0, len(X_val) - 1)

        image, mask = utils.load_tensor(X_val[idx], y_val[idx], size=size)
        mask = mask.unsqueeze(0)
        image = image.unsqueeze(0)
        vanilla_prediction = model(image)
        utils.compare_prediction(image, mask, vanilla_prediction)