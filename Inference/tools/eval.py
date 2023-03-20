import os
import sys
import time
from copy import deepcopy

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

from dehazing_sr.config import get_cfg
from dehazing_sr.engine import default_argument_parser, default_setup
from dehazing_sr.data import LoadTraining, LoadVal, LoadTest 
from dehazing_sr.architectures import RDLUF_MixS2, MixS2SR
from dehazing_sr.metrics import lpips, torch_psnr, torch_ssim
from dehazing_sr.engine import seed_everything
seed_everything(3407, deterministic=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dehazing_model_list = []
dehazing_ema_list = []
sr_model_list = []
sr_ema_list = []

args = default_argument_parser().parse_args()
for config_file in args.config_files:
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    dehazing_model = RDLUF_MixS2(cfg).to(device)
    dehazing_ema = ExponentialMovingAverage(dehazing_model.parameters(), decay=cfg.MODEL.EMA.DECAY)
    if cfg.PRETRAINED_DEHAZING_CKPT_PATH:
        print(f"===> Loading Checkpoint from {cfg.PRETRAINED_DEHAZING_CKPT_PATH}")
        save_state = torch.load(cfg.PRETRAINED_DEHAZING_CKPT_PATH)
        dehazing_model.load_state_dict(save_state['model'])
        dehazing_ema.load_state_dict(save_state['ema'])

    dehazing_model_list.append(dehazing_model)
    dehazing_ema_list.append(dehazing_ema)

    sr_model = MixS2SR(cfg).to(device)
    sr_ema = ExponentialMovingAverage(sr_model.parameters(), decay=cfg.MODEL.EMA.DECAY)

    if cfg.PRETRAINED_SR_CKPT_PATH:
        print(f"===> Loading Checkpoint from {cfg.PRETRAINED_SR_CKPT_PATH}")
        save_state = torch.load(cfg.PRETRAINED_SR_CKPT_PATH)
        sr_model.load_state_dict(save_state['model'])
        sr_ema.load_state_dict(save_state['ema'])

    sr_model_list.append(sr_model)
    sr_ema_list.append(sr_ema)


train_imgs, train_labels = LoadTraining(cfg.DATASETS.TRAIN.PATHS, debug=True)
val_imgs, val_labels = LoadVal(cfg.DATASETS.VAL.PATH)

perceptual_alex = lpips.LPIPS(net='alex').to(device)

imgs_and_labels = {
    "train" : (train_imgs, train_labels),
    "val" : (val_imgs, val_labels),
}


def single_model_single_image(
        dehazing_model, 
        dehazing_ema,
        sr_model,
        sr_ema, 
        val_img, 
    ):
    dehazing_model.eval()
    sr_model.eval()
    origin_h, origin_w, _ = val_img.shape
    if origin_h > origin_w:
        val_img = val_img.transpose(1, 0, 2)
    val_img = cv2.resize(val_img, cfg.MODEL.TEST_SIZE[::-1])

    val_img  = torch.from_numpy(val_img.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)

    if cfg.MODEL.TTA:
        inp_list = [deepcopy(val_img), deepcopy(val_img).flip((2, )), deepcopy(val_img).flip((3),), deepcopy(val_img).flip((2, 3))]
        out_list = []
        for inp_ in inp_list:
            with torch.no_grad():
                with dehazing_ema.average_parameters():
                    out, log_dict = dehazing_model(inp_)

                with sr_ema.average_parameters():
                    out = sr_model(out)
                out_list.append(out)

        out_list = [out_list[0], out_list[1].flip((2, )), out_list[2].flip((3, )), out_list[3].flip((2, 3))]
        out = (out_list[0] + out_list[1] + out_list[2] + out_list[3]) / len(out_list)
    else:
        with torch.no_grad():
            with dehazing_ema.average_parameters():                
                out, log_dict = dehazing_model(val_img)

            with sr_ema.average_parameters():
                out = sr_model(out)

    if origin_h > origin_w:
        out = out.permute(0, 1, 3, 2)        

    return out

def ensemble_eval(mode, imgs, labels):
    psnr_hr_list, ssim_hr_list, perceptual_hr_list = [], [], []
    begin = time.time()
    for i, (img, label) in enumerate(zip(imgs, labels)):
        print(f"===> image: {i}")
        val_label = torch.from_numpy(label.transpose(2, 0, 1)).to(torch.float32).to(device)
        pred_list = []
        for j, (dehazing_model, dehazing_ema, sr_model, sr_ema) in enumerate(zip(dehazing_model_list, dehazing_ema_list, sr_model_list, sr_ema_list)):
            print(f"===> model: {j}")
            pred = single_model_single_image(dehazing_model, dehazing_ema, sr_model, sr_ema, img)
            pred_list.append(pred)
        model_out = torch.mean(torch.concat(pred_list, dim=0), dim=0).unsqueeze(0).clip(0, 1)

        out = model_out.cpu().squeeze().permute(1, 2, 0).numpy() * 255
        cv2.imwrite(f"./output/Ensemble_3Stage_5Stage_7Stage_val/{i}.png", out)

        psnr_val_hr = torch_psnr(model_out[0, :, :, :], val_label)
        ssim_val_hr = torch_ssim(model_out[0, :, :, :], val_label)
        perceptual_val_hr = perceptual_alex(model_out, val_label[None, :, :, :], normalize=True)

        print("image: {}, psnr: {}, ssim: {}, perceptual: {}".format(i, psnr_val_hr, ssim_val_hr, perceptual_val_hr))
        
        psnr_hr_list.append(psnr_val_hr.detach().cpu().numpy())
        ssim_hr_list.append(ssim_val_hr.detach().cpu().numpy())
        perceptual_hr_list.append(perceptual_val_hr.detach().cpu().numpy())
    end = time.time()

    psnr_hr_mean = np.mean(np.asarray(psnr_hr_list))
    ssim_hr_mean = np.mean(np.asarray(ssim_hr_list))
    perceptual_hr_mean = np.mean(np.asarray(perceptual_hr_list))
    print('===>testing psnr = {:.2f}, ssim = {:.3f}, perceptual = {:.3f}, time: {:.2f}'.format(psnr_hr_mean, ssim_hr_mean, perceptual_hr_mean,(end - begin)))



def eval(mode, model, imgs, labels):
    psnr_hr_list, ssim_hr_list, perceptual_hr_list = [], [], []    
    dehazing_model.eval()
    sr_model.eval()
    begin = time.time()

    for i, (val_img, val_label) in enumerate(zip(imgs, labels)):
        print(f"===> {i}")
        origin_h, origin_w, _ = val_img.shape
        if origin_h > origin_w:
            val_img = val_img.transpose(1, 0, 2)
            val_label = val_label.transpose(1, 0, 2)
        origin_val_label = torch.from_numpy(val_label.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
        val_img = cv2.resize(val_img, cfg.MODEL.TEST_SIZE[::-1])
        val_label = cv2.resize(val_label, cfg.MODEL.TEST_SIZE[::-1])

        val_img  = torch.from_numpy(val_img.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
        val_label = torch.from_numpy(val_label.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)

        if cfg.MODEL.TTA:
            inp_list = [deepcopy(val_img), deepcopy(val_img).flip((2, )), deepcopy(val_img).flip((3),), deepcopy(val_img).flip((2, 3))]
            out_list = []
            for inp_ in inp_list:
                with torch.no_grad():
                    with dehazing_ema.average_parameters():
                        out, log_dict = dehazing_model(inp_)

                    with sr_ema.average_parameters():
                        out = sr_model(out)
                    out_list.append(out)

            out_list = [out_list[0], out_list[1].flip((2, )), out_list[2].flip((3, )), out_list[3].flip((2, 3))]
            out = (out_list[0] + out_list[1] + out_list[2] + out_list[3]) / len(out_list)
        else:
            with torch.no_grad():
                with dehazing_ema.average_parameters():                
                    out, log_dict = dehazing_model(val_img)

                with sr_ema.average_parameters():
                    out = sr_model(out)

        model_out = out.clip(0, 1)

        out = out.clip(0, 1).cpu().squeeze().permute(1, 2, 0).numpy() * 255
        if origin_h > origin_w:
            out = out.transpose(1, 0, 2)
        cv2.imwrite(f"./output/val/{i}.png", out)
        
        
        psnr_val_hr = torch_psnr(model_out[0, :, :, :], origin_val_label[0, :, :, :])
        ssim_val_hr = torch_ssim(model_out[0, :, :, :], origin_val_label[0, :, :, :])
        perceptual_val_hr = perceptual_alex(model_out, origin_val_label, normalize=True)
        
        psnr_hr_list.append(psnr_val_hr.detach().cpu().numpy())
        ssim_hr_list.append(ssim_val_hr.detach().cpu().numpy())
        perceptual_hr_list.append(perceptual_val_hr.detach().cpu().numpy())

    end = time.time()

    psnr_hr_mean = np.mean(np.asarray(psnr_hr_list))
    ssim_hr_mean = np.mean(np.asarray(ssim_hr_list))
    perceptual_hr_mean = np.mean(np.asarray(perceptual_hr_list))
    print('===>HR: testing psnr = {:.2f}, ssim = {:.3f}, perceptual = {:.3f}, time: {:.2f}'.format(psnr_hr_mean, ssim_hr_mean, perceptual_hr_mean,(end - begin)))

    dehazing_model.train()
    sr_model.train()
    
    return psnr_hr_list, ssim_hr_list, perceptual_hr_list, psnr_hr_mean, ssim_hr_mean, perceptual_hr_mean


def main():
    mode = "val"
    # eval(mode, *(imgs_and_labels[mode]))
    ensemble_eval(mode, *(imgs_and_labels[mode]))

if __name__ == '__main__':
    main()