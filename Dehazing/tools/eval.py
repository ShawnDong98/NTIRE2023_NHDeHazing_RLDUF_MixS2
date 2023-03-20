import os
import sys
import time

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision.models import vgg16
from torchvision.utils import make_grid
from torch_ema import ExponentialMovingAverage

import cv2
import numpy as np
from tqdm import tqdm

from dehazing.config import get_cfg
from dehazing.engine import default_argument_parser, default_setup
from dehazing.data import DehazingTrainDataset, LoadTraining, LoadVal, LoadTest  
from dehazing.architectures import RDLUF_MixS2, Discriminator
from dehazing.metrics import lpips, torch_psnr, torch_ssim
from dehazing.engine import seed_everything
seed_everything(3407, deterministic=True)

args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
train_imgs, train_labels = LoadTraining(["../datasets/NTIRE2023/Dehazing/train_/"], debug=False)
val_imgs, val_labels = LoadVal(cfg.DATASETS.VAL.PATH)
test_imgs = LoadTest(cfg.DATASETS.TEST.PATH)

imgs_and_labels = {
    "train" : (train_imgs, train_labels),
    "val" : (val_imgs, val_labels),
    "test" : (test_imgs, None) 
}

model = RDLUF_MixS2(cfg).to(device)

ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)

if cfg.PRETRAINED_CKPT_PATH:
    print(f"===> Loading Checkpoint from {cfg.PRETRAINED_CKPT_PATH}")
    save_state = torch.load(cfg.PRETRAINED_CKPT_PATH)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])

perceptual_alex = lpips.LPIPS(net='alex').to(device)



def eval(mode, imgs, labels):
    psnr_lr_list, ssim_lr_list, perceptual_lr_list = [], [], []
    psnr_hr_list, ssim_hr_list, perceptual_hr_list = [], [], []    
    model.eval()
    begin = time.time()
    inp = []
    pred = []
    gt = []

    for i, (val_img, val_label) in enumerate(zip(imgs, labels)):
        origin_h, origin_w, _ = val_img.shape
        if origin_h > origin_w:
            val_img = val_img.transpose(1, 0, 2)
            val_label = val_label.transpose(1, 0, 2)
        origin_val_label = torch.from_numpy(val_label.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
        val_img = cv2.resize(val_img, cfg.MODEL.TEST_SIZE[::-1])
        val_label = cv2.resize(val_label, cfg.MODEL.TEST_SIZE[::-1])

        val_img  = torch.from_numpy(val_img.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
        val_label = torch.from_numpy(val_label.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
        with torch.no_grad():
            with ema.average_parameters():
                out, log_dict = model(val_img)
                model_out = out

                inp.append(val_img)
                pred.append(model_out)
                gt.append(val_label)

            out = out.clip(0, 1).cpu().squeeze().permute(1, 2, 0).numpy() * 255
            if origin_h > origin_w:
                out = out.transpose(1, 0, 2)
            cv2.imwrite(f"../datasets/NTIRE2023/SR_9Stage/{mode}/LR/" + str(i+1).zfill(2) + ".png", out)
            
            psnr_val_lr = torch_psnr(model_out[0, :, :, :], val_label[0, :, :, :])
            ssim_val_lr = torch_ssim(model_out[0, :, :, :], val_label[0, :, :, :])
            perceptual_val_lr = perceptual_alex(model_out, val_label, normalize=True)
            
            model_out = F.interpolate(model_out,  size=(4000, 6000), mode='bilinear')
            psnr_val_hr = torch_psnr(model_out[0, :, :, :], origin_val_label[0, :, :, :])
            ssim_val_hr = torch_ssim(model_out[0, :, :, :], origin_val_label[0, :, :, :])
            perceptual_val_hr = perceptual_alex(model_out, origin_val_label, normalize=True)
            
            psnr_lr_list.append(psnr_val_lr.detach().cpu().numpy())
            ssim_lr_list.append(ssim_val_lr.detach().cpu().numpy())
            perceptual_lr_list.append(perceptual_val_lr.detach().cpu().numpy())

            psnr_hr_list.append(psnr_val_hr.detach().cpu().numpy())
            ssim_hr_list.append(ssim_val_hr.detach().cpu().numpy())
            perceptual_hr_list.append(perceptual_val_hr.detach().cpu().numpy())

    end = time.time()

    psnr_lr_mean = np.mean(np.asarray(psnr_lr_list))
    ssim_lr_mean = np.mean(np.asarray(ssim_lr_list))
    perceptual_lr_mean = np.mean(np.asarray(perceptual_lr_list))
    psnr_hr_mean = np.mean(np.asarray(psnr_hr_list))
    ssim_hr_mean = np.mean(np.asarray(ssim_hr_list))
    perceptual_hr_mean = np.mean(np.asarray(perceptual_hr_list))
    print('===>LR: testing psnr = {:.2f}, ssim = {:.3f}, perceptual = {:.3f}, time: {:.2f}'.format(psnr_lr_mean, ssim_lr_mean, perceptual_lr_mean,(end - begin)))
    print('===>HR: testing psnr = {:.2f}, ssim = {:.3f}, perceptual = {:.3f}, time: {:.2f}'.format(psnr_hr_mean, ssim_hr_mean, perceptual_hr_mean,(end - begin)))

    model.train()
    
    return psnr_lr_list, ssim_lr_list, perceptual_lr_list, psnr_lr_mean, ssim_lr_mean, perceptual_lr_mean, \
        psnr_hr_list, ssim_hr_list, perceptual_hr_list, psnr_hr_mean, ssim_hr_mean, perceptual_hr_mean

def test():
    model.eval()
    for i, test_img in enumerate(test_imgs):
        origin_h, origin_w, _ = test_img.shape
        if origin_h > origin_w:
            test_img = test_img.transpose(1, 0, 2)

        test_img = cv2.resize(test_img, cfg.MODEL.TEST_SIZE[::-1])

        test_img  = torch.from_numpy(test_img.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
        with torch.no_grad():
            with ema.average_parameters():
                out, log_dict = model(test_img)
        pred = out.clip(0, 1).cpu().squeeze().permute(1, 2, 0).numpy() * 255
        if origin_h > origin_w:
            pred = pred.transpose(1, 0, 2)
        cv2.imwrite(os.path.join("../datasets/NTIRE2023/SR_9Stage/test_b/", "LR", f"{i}.png"), pred)                

    end = time.time()
    model.train()
    return pred

def main():
    mode = "train"
    if mode in ['train', 'val']:
        eval(mode, *(imgs_and_labels[mode]))
    else:
        test()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()