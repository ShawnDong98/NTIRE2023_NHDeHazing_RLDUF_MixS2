import os
import sys
import time

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import FlopCountAnalysis


from sr.config import get_cfg
from sr.engine import default_argument_parser, default_setup
from sr.architectures import MixS2SR, Discriminator


args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()
logger, writer, output_dir = default_setup(cfg, args)

device = torch.device("cpu")

model = MixS2SR(cfg).to(device)
inp = torch.zeros((1, 3, 2000, 3000)).to(device)

flops = FlopCountAnalysis(model, inp)
print(f'GMac:{flops.total()/(1024*1024*1024)}')
print("Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))