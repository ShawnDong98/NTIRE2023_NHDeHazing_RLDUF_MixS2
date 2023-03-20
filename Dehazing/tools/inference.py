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
from dehazing.data import DehazingTrainDataset, LoadVal, LoadTest  
from dehazing.architectures import RDLUF_MixS2, Discriminator
from dehazing.metrics import lpips, torch_psnr, torch_ssim
from dehazing.engine import seed_everything
seed_everything(3407, deterministic=True)

args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_imgs, val_labels = LoadVal(cfg.DATASETS.VAL.PATH)
test_imgs = LoadTest(cfg.DATASETS.TEST.PATH)

model = RDLUF_MixS2(cfg).to(device)

ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)

if cfg.PRETRAINED_CKPT_PATH:
    print(f"===> Loading Checkpoint from {cfg.PRETRAINED_CKPT_PATH}")
    save_state = torch.load(cfg.PRETRAINED_CKPT_PATH)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])

perceptual_alex = lpips.LPIPS(net='alex').to(device)


def test():
    model.eval()
    begin = time.time()
    inp = []
    pred = []
    for k, test_img in enumerate(test_imgs):
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
        pred = cv2.resize(pred, (origin_w, origin_h))
        canvas = np.ones((origin_h, origin_w, 1), dtype=np.uint8) * 255
        pred = np.concatenate((pred, canvas), axis=2)
        cv2.imwrite(os.path.join("./output/", "3Stage_Origin2000_TV_epoch115_TestB", f"{k}.png"), pred)                
    end = time.time()
    model.train()
    print('===>testing time: {:.2f}'.format((end - begin)))


if __name__ == "__main__":
    test()