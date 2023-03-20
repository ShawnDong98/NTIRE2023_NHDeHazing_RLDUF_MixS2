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
from torch_ema import ExponentialMovingAverage


from dehazing_sr.config import get_cfg
from dehazing_sr.engine import default_argument_parser, default_setup
from dehazing_sr.architectures import RDLUF_MixS2, MixS2SR, Discriminator


args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_files[0])
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dehazing_model = RDLUF_MixS2(cfg).to(device)
dehazing_model.eval()
sr_model = MixS2SR(cfg).to(device)
sr_model.eval()


inp = torch.zeros((1, 3, 2000, 3000)).to(device)

begin = time.time()
with torch.no_grad():
    out, _ = dehazing_model(inp)
    out = sr_model(out)
end = time.time()

print("Comsumed Time: ", end - begin)