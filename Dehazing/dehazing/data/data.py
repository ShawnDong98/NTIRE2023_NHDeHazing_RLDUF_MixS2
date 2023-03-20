import os
import random

import torch
from torch.utils.data import Dataset, DataLoader

import cv2
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

class DehazingTrainDataset(Dataset):
    def __init__(
        self, 
        cfg,
        crop_size=(200, 300)
    ):
        super().__init__()
        self.iteration = cfg.DATASETS.TRAIN.ITERATION if not cfg.DEBUG else 10
        self.crop_size = crop_size
        self.origin_size = cfg.DATASETS.TRAIN.ORIGIN_SIZE
        self.augment = cfg.DATASETS.TRAIN.AUGMENT


        self.imgs, self.labels = LoadTraining(cfg.DATASETS.TRAIN.PATHS, cfg.DEBUG)

        assert len(self.imgs) == len(self.labels), "The number of images and labels are not equal."
        self.len_images = len(self.imgs)

    def __getitem__(self, idx):
        if self.augment:
            flag = random.randint(0, 2)
            if flag > 0:
                index = np.random.randint(0, self.len_images-1)
                img, label = self.imgs[index], self.labels[index]
                processed_image = np.zeros((self.crop_size[0], self.crop_size[1], 3), dtype=np.float32)
                processed_label = np.zeros((self.crop_size[0], self.crop_size[1], 3), dtype=np.float32)
        
                origin_h, origin_w, _ = img.shape
                if origin_h > origin_w:
                    img = np.transpose(img, (1, 0, 2))
                    label = np.transpose(label, (1, 0, 2))
                    origin_h, origin_w = origin_w, origin_h
                scale_factor = origin_h / self.origin_size[0]
                h = int(origin_h / scale_factor)
                w = int(origin_w / scale_factor)
                
                img = cv2.resize(img,  (w, h), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label,  (w, h), interpolation=cv2.INTER_LINEAR)
                x_index = np.random.randint(0, h - self.crop_size[0])
                y_index = np.random.randint(0, w - self.crop_size[1])
                processed_image = img[x_index:x_index + self.crop_size[0], y_index:y_index + self.crop_size[1], :]
                processed_label = label[x_index:x_index + self.crop_size[0], y_index:y_index + self.crop_size[1], :]

                processed_image = torch.from_numpy(np.transpose(processed_image, (2, 0, 1)))
                processed_label = torch.from_numpy(np.transpose(processed_label, (2, 0, 1)))

                processed_image, processed_label = augment_1(processed_image, processed_label)
            else:
                processed_image = np.zeros((4, self.crop_size[0]//2, self.crop_size[1]//2, 3), dtype=np.float32)
                processed_label = np.zeros((4, self.crop_size[0]//2, self.crop_size[1]//2, 3), dtype=np.float32)
                sample_list = np.random.randint(0, self.len_images, 4)
                for j in range(4):
                    origin_h, origin_w, _ = self.imgs[sample_list[j]].shape
                    if origin_h > origin_w:
                        self.imgs[sample_list[j]] = np.transpose(self.imgs[sample_list[j]], (1, 0, 2))
                        self.labels[sample_list[j]] = np.transpose(self.labels[sample_list[j]], (1, 0, 2))
                        origin_h, origin_w = origin_w, origin_h
                    scale_factor = origin_h / self.origin_size[0]
                    h = int(origin_h / scale_factor)
                    w = int(origin_w / scale_factor)
                    self.imgs[sample_list[j]] = cv2.resize(self.imgs[sample_list[j]], (w, h))
                    self.labels[sample_list[j]] = cv2.resize(self.labels[sample_list[j]], (w, h))
                    x_index = np.random.randint(0, h-self.crop_size[0]//2)
                    y_index = np.random.randint(0, w-self.crop_size[1]//2)
                    processed_image[j] = self.imgs[sample_list[j]][x_index:x_index+self.crop_size[0]//2,y_index:y_index+self.crop_size[1]//2,:]
                    processed_label[j] = self.labels[sample_list[j]][x_index:x_index+self.crop_size[0]//2,y_index:y_index+self.crop_size[1]//2,:]
                processed_image = torch.from_numpy(np.transpose(processed_image, (0, 3, 1, 2)))  # [4,28,128,128]
                processed_label = torch.from_numpy(np.transpose(processed_label, (0, 3, 1, 2)))  # [4,28,128,128]
                processed_image, processed_label = augment_2(processed_image, processed_label, crop_size=self.crop_size)
            
        return processed_image, processed_label
    
    def __len__(self):
        return self.iteration



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


def augment_2(generate_image, generate_label, crop_size=(512, 768)):
    c, h, w = generate_image.shape[1], crop_size[0], crop_size[1]
    divid_point_h = crop_size[0] // 2 
    divid_point_w = crop_size[1] // 2
    output_img = torch.zeros(c,h,w)
    output_label = torch.zeros(c,h,w)
    output_img[:, :divid_point_h, :divid_point_w] = generate_image[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_image[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_image[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_image[3]
    output_label[:, :divid_point_h, :divid_point_w] = generate_label[0]
    output_label[:, :divid_point_h, divid_point_w:] = generate_label[1]
    output_label[:, divid_point_h:, :divid_point_w] = generate_label[2]
    output_label[:, divid_point_h:, divid_point_w:] = generate_label[3]
    return output_img, output_label


if __name__ == "__main__":
    cfg = Box(
        {
            "DEBUG": True,
            "DATASETS": 
            {
                "TRAIN": 
                {
                    "PATHS" : ["../../../datasets/NTIRE2023/Dehazing/train/"],
                    "ITERATION": 1000,
                    "ORIGIN_SIZE": (2000, 3000),
                    "AUGMENT": True,
                }
            },
            "MODEL": 
            {
                "TEST_SIZE": (2000, 3000)
            }
        }
    )
    dataset = DehazingTrainDataset(cfg, crop_size=(400, 600))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (img, label) in enumerate(loader):
        print(img.shape, label.shape)
        cv2.imshow("image", img.squeeze().cpu().numpy().transpose(1, 2, 0))
        cv2.imshow("label", label.squeeze().cpu().numpy().transpose(1, 2, 0))
        cv2.waitKey(0)

