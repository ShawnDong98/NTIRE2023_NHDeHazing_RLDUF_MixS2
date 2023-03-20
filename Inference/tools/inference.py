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

print(args)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

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


test_imgs = LoadTest(cfg.DATASETS.TEST.PATH)


perceptual_alex = lpips.LPIPS(net='alex').to(device)


def single_model_single_image(
        dehazing_model, 
        dehazing_ema,
        sr_model,
        sr_ema, 
        img, 
    ):
    dehazing_model.eval()
    sr_model.eval()
    origin_h, origin_w, _ = img.shape
    if origin_h > origin_w:
        img = img.transpose(1, 0, 2)
    img = cv2.resize(img, cfg.MODEL.TEST_SIZE[::-1])

    img  = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)

    if cfg.MODEL.TTA:
        inp_list = [deepcopy(img), deepcopy(img).flip((2, )), deepcopy(img).flip((3),), deepcopy(img).flip((2, 3))]
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
                out, log_dict = dehazing_model(img)

            with sr_ema.average_parameters():
                out = sr_model(out)

    if origin_h > origin_w:
        out = out.permute(0, 1, 3, 2)        

    return out

def ensemble_test(imgs):
    begin = time.time()
    for i, img in enumerate(imgs):
        print(f"===> image: {i}")
        pred_list = []
        for j, (dehazing_model, dehazing_ema, sr_model, sr_ema) in enumerate(zip(dehazing_model_list, dehazing_ema_list, sr_model_list, sr_ema_list)):
            print(f"===> model: {j}")
            pred = single_model_single_image(dehazing_model, dehazing_ema, sr_model, sr_ema, img)
            pred_list.append(pred)
        model_out = torch.mean(torch.concat(pred_list, dim=0), dim=0).unsqueeze(0).clip(0, 1)

        out = model_out.cpu().squeeze().permute(1, 2, 0).numpy() * 255
        h, w, c = out.shape
        canvas = np.ones((h, w, 1), dtype=np.uint8) * 255
        out = np.concatenate((out, canvas), axis=2)

        cv2.imwrite(os.path.join(args.output_dir, f"{i}.png"), out)

    end = time.time()

    print('===>testing time: {:.2f}'.format((end - begin)))


def test():
    dehazing_model.eval()
    sr_model.eval()
    for k, test_img in enumerate(test_imgs):
        print("k: ", k)
        origin_h, origin_w, _ = test_img.shape
        if origin_h > origin_w:
            test_img = test_img.transpose(1, 0, 2)

        test_img = cv2.resize(test_img, tuple(cfg.MODEL.TEST_SIZE[::-1]))

        test_img  = torch.from_numpy(test_img.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)

        if cfg.MODEL.TTA:
            inp_list = [deepcopy(test_img), deepcopy(test_img).flip((2, )), deepcopy(test_img).flip((3),), deepcopy(test_img).flip((2, 3))]
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
                    out, log_dict = dehazing_model(test_img)

                with sr_ema.average_parameters():
                    out = sr_model(out)

        pred = out.clip(0, 1).cpu().squeeze().permute(1, 2, 0).numpy() * 255
        
        output_size = (cfg.MODEL.TEST_SIZE[0] * cfg.MODEL.SR.UP_SCALE, cfg.MODEL.TEST_SIZE[1] * cfg.MODEL.SR.UP_SCALE)
        if origin_h > origin_w:
            pred = pred.transpose(1, 0, 2)
            output_size = output_size[::-1]
        
        canvas = np.ones((output_size[0], output_size[1], 1), dtype=np.uint8) * 255
        pred = np.concatenate((pred, canvas), axis=2)
        cv2.imwrite(os.path.join("./output/", "Dehazing_5Stage_E145_SR253_TTA", f"{k}.png"), pred)  





if __name__ == '__main__':
    # test()
    ensemble_test(test_imgs)    
