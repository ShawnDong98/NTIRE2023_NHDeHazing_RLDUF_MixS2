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

from sr.config import get_cfg
from sr.engine import default_argument_parser, default_setup
from sr.data import LoadTraining, LoadVal, LoadTest, SRTrainDataset
from sr.architectures import MixS2SR, Discriminator
from sr.utils.schedulers import get_cosine_schedule_with_warmup
from sr.losses import CharbonnierLoss, LossNetwork, ms_ssim, TVLoss
from sr.metrics import lpips, torch_psnr, torch_ssim
from sr.utils import checkpoint

args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()
logger, writer, output_dir = default_setup(cfg, args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


crop_phase = 0
val_imgs, val_labels = LoadVal(cfg.DATASETS.VAL.PATH)
test_imgs = LoadTest(cfg.DATASETS.TEST.PATH)

model = MixS2SR(cfg).to(device)

if cfg.LOSS.GAN_LOSS:
    DNet = Discriminator(inp_nc=cfg.MODEL.DISCRIMINATOR.IN_DIM).to(device)
    DNet = DNet.to(device)
    D_optim = torch.optim.Adam(DNet.parameters(), lr=cfg.MODEL.DISCRIMINATOR.LR)


ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)

# optimizing
optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LR, betas=(0.9, 0.999))


scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(np.floor(cfg.DATASETS.TRAIN.ITERATION / cfg.DATALOADER.BATCH_SIZE[-1])), 
    num_training_steps=int(np.floor(cfg.DATASETS.TRAIN.ITERATION / cfg.DATALOADER.BATCH_SIZE[-1])) * cfg.OPTIMIZER.MAX_EPOCH, 
    eta_min=1e-6)

start_epoch = 0

if cfg.RESUME_CKPT_PATH:
    print(f"===> Loading Checkpoint from {cfg.RESUME_CKPT_PATH}")
    save_state = torch.load(cfg.RESUME_CKPT_PATH)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])
    optimizer.load_state_dict(save_state['optimizer'])
    scheduler.load_state_dict(save_state['scheduler'])
    start_epoch = save_state['epoch']

l1_loss = CharbonnierLoss().to(device)

if cfg.LOSS.TV_LOSS:
    tv_loss = TVLoss().to(device)

if cfg.LOSS.SSIM_LOSS:
    ssim_loss = ms_ssim

if cfg.LOSS.VGG_LOSS:
    vgg_model = vgg16(pretrained=True)
    vgg_model = vgg_model.features[:16].to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    vgg_loss = LossNetwork(vgg_model)
    vgg_loss.eval()


perceptual_alex = lpips.LPIPS(net='alex').to(device)


def train(epoch, train_loader):
    model.train()
    if cfg.LOSS.GAN_LOSS:
        DNet.train()
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(cfg.DATASETS.TRAIN.ITERATION / train_loader.batch_size))
    train_tqdm = tqdm(range(batch_num)[:5])  if cfg.DEBUG else tqdm(range(batch_num))

    loss_dict = {}
    for i in train_tqdm:
        data_time = time.time()
        try:
            images_batch, labels_batch = next(data_iter)
        except:
            data_iter = iter(train_loader)
            images_batch, labels_batch = next(data_iter)
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)
        data_time = time.time() - data_time

        model_time = time.time()
        model_out = model(images_batch)
        model_time = time.time() - model_time

        loss = 0
        loss_l1 = l1_loss(model_out, labels_batch)
        loss_dict['loss_l1'] = f"{loss_l1.item():.4f}"
        loss += loss_l1

        if cfg.LOSS.TV_LOSS:
            loss_tv = 0.01 * tv_loss(model_out)
            loss_dict['loss_tv'] = f"{loss_tv.item():.4f}"
            loss += loss_tv

        if cfg.LOSS.SSIM_LOSS:
            loss_ssim =  0.2 * (1 - ssim_loss(model_out, labels_batch, data_range=1.0))
            loss_dict['loss_ssim'] = f"{loss_ssim.item():.4f}"
            loss += loss_ssim
        if cfg.LOSS.VGG_LOSS:
            loss_vgg = 0.01 * (vgg_loss(model_out, labels_batch))
            loss_dict['loss_vgg'] = f"{loss_vgg.item():.4f}"
            loss += loss_vgg
        if cfg.LOSS.GAN_LOSS:
            real_out = DNet(labels_batch).mean()
            fake_out = DNet(model_out.detach()).mean()
            D_loss = 1 - real_out + fake_out
            D_loss.backward()

            fake_out = DNet(model_out).mean()
            gan_loss = 0.0005 * (torch.mean(1 - fake_out))
            loss_dict['gan_loss'] = f"{gan_loss.item():.4f}"
            loss += gan_loss
        

        loss.backward()
        if cfg.OPTIMIZER.GRAD_CLIP:
            clip_grad_norm_(model.parameters(), max_norm=0.2)

        optimizer.step()
        if cfg.LOSS.GAN_LOSS:
            D_optim.step()
        optimizer.zero_grad()
        if cfg.LOSS.GAN_LOSS:
            D_optim.zero_grad()
        ema.update()
        loss_dict['data_time'] = data_time
        loss_dict['model_time'] = model_time
        train_tqdm.set_postfix(loss_dict)
        epoch_loss += loss.data
        writer.add_scalar('LR/train',optimizer.state_dict()['param_groups'][0]['lr'], epoch * batch_num + i)
        scheduler.step()
    end = time.time()
    train_loss = epoch_loss / batch_num
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, train_loss, (end - begin)))
    return train_loss


def eval(epoch):
    psnr_lr_list, ssim_lr_list, perceptual_lr_list = [], [], []
    model.eval()
    begin = time.time()
    inp = []
    pred = []
    gt = []

    for val_img, val_label in zip(val_imgs, val_labels):
        origin_h, origin_w, _ = val_img.shape
        if origin_h > origin_w:
            val_img = val_img.transpose(1, 0, 2)
            val_label = val_label.transpose(1, 0, 2)

        val_img  = torch.from_numpy(val_img.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
        val_label = torch.from_numpy(val_label.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
        with torch.no_grad():
            with ema.average_parameters():
                out = model(val_img)
                model_out = out

                inp.append(val_img)
                pred.append(model_out)
                gt.append(val_label)
            
            psnr_val_lr = torch_psnr(model_out[0, :, :, :], val_label[0, :, :, :])
            ssim_val_lr = torch_ssim(model_out[0, :, :, :], val_label[0, :, :, :])
            perceptual_val_lr = perceptual_alex(model_out, val_label, normalize=True)
            
            
            psnr_lr_list.append(psnr_val_lr.detach().cpu().numpy())
            ssim_lr_list.append(ssim_val_lr.detach().cpu().numpy())
            perceptual_lr_list.append(perceptual_val_lr.detach().cpu().numpy())


    end = time.time()

    inp = F.interpolate(torch.cat(inp, dim = 0), size=(128, 192), mode='bilinear')
    pred = F.interpolate(torch.cat(pred, dim = 0), size=(128, 192), mode='bilinear')
    gt = F.interpolate(torch.cat(gt, dim = 0), size=(128, 192), mode='bilinear')
    pred_gt = torch.cat([inp, pred, gt], dim = 0).clip(0, 1)
    grid = make_grid(pred_gt, nrow=5)
    writer.add_image('images/val', grid, epoch)
    psnr_lr_mean = np.mean(np.asarray(psnr_lr_list))
    ssim_lr_mean = np.mean(np.asarray(ssim_lr_list))
    perceptual_lr_mean = np.mean(np.asarray(perceptual_lr_list))
    logger.info('===>LR: testing psnr = {:.2f}, ssim = {:.3f}, perceptual = {:.3f}, time: {:.2f}'.format(psnr_lr_mean, ssim_lr_mean, perceptual_lr_mean,(end - begin)))
    model.train()
    
    return psnr_lr_list, ssim_lr_list, perceptual_lr_list, psnr_lr_mean, ssim_lr_mean, perceptual_lr_mean


def test(epoch):
    model.eval()
    begin = time.time()
    inp = []
    pred = []
    for test_img in test_imgs:
        origin_h, origin_w, _ = test_img.shape
        if origin_h > origin_w:
            test_img = test_img.transpose(1, 0, 2)

        test_img  = torch.from_numpy(test_img.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
        with torch.no_grad():
            with ema.average_parameters():
                out = model(test_img)
                inp.append(test_img)
                pred.append(out)

    end = time.time()
    inp = torch.cat(inp, dim = 0)
    pred = torch.cat(pred, dim = 0)
    inp = F.interpolate(inp, size=(128, 192), mode='bilinear')
    pred = F.interpolate(pred, size=(128, 192), mode='bilinear')
    pred_gt = torch.cat([inp, pred], dim = 0).clip(0, 1)
    grid = make_grid(pred_gt, nrow=5)
    writer.add_image('images/test', grid, epoch)
    model.train()
    logger.info('===> Epoch {}: testing time: {:.2f}'.format(epoch, (end - begin)))
    model.train()
    return grid

def main():
    psnr_max = 0
    crop_phase = 0
    cnt = 0
    dataset = SRTrainDataset(cfg, crop_size=cfg.DATASETS.TRAIN.CROP_SIZE[0])
    for epoch in range(start_epoch, cfg.OPTIMIZER.MAX_EPOCH):
        if cnt >= 5:
            crop_phase += 1
            crop_phase = min(len(cfg.DATASETS.TRAIN.CROP_SIZE)-1, crop_phase)
            cnt = 0

        logger.info(f"===> Crop Phase: {crop_phase}")
        crop_size = cfg.DATASETS.TRAIN.CROP_SIZE[crop_phase]
        batch_size = cfg.DATALOADER.BATCH_SIZE[crop_phase]
        train_loader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
            pin_memory = False,
            drop_last = False
        )
        setattr(train_loader.dataset, 'crop_size', crop_size)
        
        train_loss = train(epoch, train_loader)
        torch.cuda.empty_cache()
        psnr_lr_all, ssim_lr_all, perceptual_lr_all, psnr_lr_mean, ssim_lr_mean, perceptual_lr_mean = eval(epoch)
        _ = test(epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('PSNR_LR/val', psnr_lr_mean, epoch)
        writer.add_scalar('SSIM_LR/val', ssim_lr_mean, epoch)
        writer.add_scalar('Perceptual_LR/val', perceptual_lr_mean, epoch)
        if psnr_lr_mean > psnr_max:
            cnt = 0
            psnr_max = psnr_lr_mean
            checkpoint(model, ema, optimizer, scheduler, epoch, output_dir, logger)
        else:
            cnt += 1
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()