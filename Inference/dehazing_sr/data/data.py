import os
import logging
import random

import torch
from torch.utils.data import Dataset, DataLoader

import cv2
from PIL import Image
import numpy as np
from glob import glob

from box import Box


def LoadTraining(paths, debug=False):
    assert isinstance(paths, list), "paths must be a list"
    imgs = []
    labels = []
    scene_list = []
    label_list = []
    for path in paths:
        scene_list.extend(glob(os.path.join(path, "images", "*")))
        label_list.extend(glob(os.path.join(path, "labels", "*")))
    scene_list.sort()
    label_list.sort()
    print('training sences:', len(scene_list))
    for scene_path, label_path in zip(scene_list, label_list) if not debug else zip(scene_list[:5], label_list[:5]):
        assert scene_path.split("/")[-1][:2] == label_path.split("/")[-1][:2], scene_path + " " + label_path
        img = cv2.imread(scene_path) / 255.
        label = cv2.imread(label_path) / 255.
        img = img.astype(np.float32)
        label = label.astype(np.float32)
        imgs.append(img)
        labels.append(label)
        print('Sence {} is loaded.'.format(scene_path.split('/')[-1]))
    return imgs, labels



def LoadVal(path_val):
    images = []
    labels = []
    scene_list = os.listdir(os.path.join(path_val, "images"))
    scene_list.sort()
    for i in range(len(scene_list)):
        scene_path = os.path.join(path_val, "images", scene_list[i])
        label_path = os.path.join(path_val, "labels", scene_list[i])
        img = cv2.imread(scene_path) / 255.
        label = cv2.imread(label_path) / 255.
        images.append(img)
        labels.append(label)

    return images, labels

def LoadTest(path_test):
    images = []
    scene_list = os.listdir(os.path.join(path_test, "images"))
    scene_list.sort()
    for i in range(len(scene_list)):
        scene_path = os.path.join(path_test, "images", scene_list[i])
        img = cv2.imread(scene_path) / 255.
        images.append(img)
    return images



def augment_1(image, label):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random vertical Flip
    for j in range(vFlip):
        image = torch.flip(image, dims=(2,))
        label = torch.flip(label, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        image = torch.flip(image, dims=(1,))
        label = torch.flip(label, dims=(1,))
    return image, label


def augment_2(generate_image, generate_label, crop_size=(512, 768), scale_factor=4):
    c, h, w = generate_image.shape[1], crop_size[0], crop_size[1]
    divid_point_h = crop_size[0] // 2 
    divid_point_w = crop_size[1] // 2
    output_img = torch.zeros(c,h,w)
    output_label = torch.zeros(c,h * scale_factor,w * scale_factor)
    output_img[:, :divid_point_h, :divid_point_w] = generate_image[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_image[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_image[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_image[3]
    output_label[:, :divid_point_h * scale_factor, :divid_point_w * scale_factor] = generate_label[0]
    output_label[:, :divid_point_h * scale_factor, divid_point_w * scale_factor:] = generate_label[1]
    output_label[:, divid_point_h * scale_factor:, :divid_point_w * scale_factor] = generate_label[2]
    output_label[:, divid_point_h * scale_factor:, divid_point_w * scale_factor:] = generate_label[3]
    return output_img, output_label


if __name__ == '__main__':
    cfg = Box(
        {
            "DEBUG": True,
            "DATASETS": 
            {
                "TRAIN": 
                {
                    "PATHS" : ["../../../datasets/NTIRE2023/SR_PG/train/"],
                    "ITERATION": 1000,
                    "CROP_SIZE": (200, 300),
                    "AUGMENT": True,
                }
            },
            "MODEL": 
            {
                "TEST_SIZE": (2000, 3000),
                "SR": {
                    "UP_SCALE": 2
        
                }
            }
        }
    )
    dataset = SRTrainDataset(cfg, crop_size=cfg.DATASETS.TRAIN.CROP_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (img, label) in enumerate(loader):
        print(img.shape, label.shape)
        cv2.imshow("image", img.squeeze().cpu().numpy().transpose(1, 2, 0))
        cv2.imshow("label", label.squeeze().cpu().numpy().transpose(1, 2, 0))
        cv2.waitKey(0)