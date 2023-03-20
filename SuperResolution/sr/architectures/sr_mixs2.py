from collections import defaultdict
import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

import numbers

ACT_FN = {
    'gelu': nn.GELU(),
    'relu' : nn.ReLU(),
    'lrelu' : nn.LeakyReLU(),
}


def DWConv(dim, kernel_size, stride, padding, bias=False):
    return nn.Conv2d(dim, dim, kernel_size, stride, padding, bias=bias, groups=dim)

def PWConv(in_dim, out_dim, bias=False):
    return nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias)

def DWPWConv(in_dim, out_dim, kernel_size, stride, padding, bias=False, act_fn_name="gelu"):
    return nn.Sequential(
        DWConv(in_dim, in_dim, kernel_size, stride, padding, bias),
        ACT_FN[act_fn_name],
        PWConv(in_dim, out_dim, bias)
    )


class BlockInteraction(nn.Module):
    def __init__(self, in_channel, out_channel, act_fn_name="gelu", bias=False):
        super(BlockInteraction, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias),
            ACT_FN[act_fn_name],
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=bias)
        )
       
    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)
    


class StageInteraction(nn.Module):
    def __init__(self, dim, act_fn_name="lrelu", bias=False):
        super().__init__()
        self.st_inter_enc = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.st_inter_dec = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.act_fn = ACT_FN[act_fn_name]
        self.phi = DWConv(dim, 3, 1, 1, bias=bias)
        self.gamma = DWConv(dim, 3, 1, 1, bias=bias)

    def forward(self, inp, pre_enc, pre_dec):
        out = self.st_inter_enc(pre_enc) + self.st_inter_dec(pre_dec)
        skip = self.act_fn(out)
        phi = torch.sigmoid(self.phi(skip))
        gamma = self.gamma(skip)

        out = phi * inp + gamma

        return out


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        return x + self.module(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class SpectralBranch(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.opt = opt
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.LayerNorm_type = LayerNorm_type

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        

    def forward(self, x, spatial_interaction=None):
        b,c,h,w = x.shape
        x = self.norm(x)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        if spatial_interaction is not None:
            q = q * spatial_interaction
            k = k * spatial_interaction
            v = v * spatial_interaction


        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (b, c, h, w) -> (b, head, c_, h * w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out) # (b, c, h, w)
        return out
    

class BasicConv2d(nn.Module):
    def __init__(self, 
                 in_planes, 
                 out_planes, 
                 kernel_size, 
                 stride, 
                 groups = 1, 
                 padding = 0, 
                 bias = False,
                 act_fn_name = "gelu",
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.act_fn = ACT_FN[act_fn_name]

    def forward(self, x):
        x = self.conv(x)
        x = self.act_fn(x)
        return x
    

class DW_Inception(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim,
                 bias=False
    ):
        super(DW_Inception, self).__init__()
        self.branch0 = BasicConv2d(in_dim, out_dim // 4, kernel_size=1, stride=1, bias=bias)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_dim, out_dim // 6, kernel_size=1, stride=1, bias=bias),
            BasicConv2d(out_dim // 6, out_dim // 6, kernel_size=3, stride=1, groups=out_dim // 6, padding=1, bias=bias),
            BasicConv2d(out_dim // 6, out_dim // 4, kernel_size=1, stride=1, bias=bias)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_dim, out_dim // 6, kernel_size=1, stride=1, bias=bias),
            BasicConv2d(out_dim // 6, out_dim // 6, kernel_size=3, stride=1, groups=out_dim // 6, padding=1, bias=bias),
            BasicConv2d(out_dim // 6, out_dim // 4, kernel_size=1, stride=1, bias=bias),
            BasicConv2d(out_dim // 4, out_dim // 4, kernel_size=3, stride=1, groups=out_dim // 4, padding=1, bias=bias),
            BasicConv2d(out_dim // 4, out_dim // 4, kernel_size=1, stride=1, bias=bias)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_dim, out_dim//4, kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class SpatialBranch(nn.Module):
    def __init__(self,
                 dim,  
                 DW_Expand=2, 
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.norm = LayerNorm(dim, LayerNorm_type = LayerNorm_type)
        self.inception = DW_Inception(
            in_dim=dim, 
            out_dim=dim*DW_Expand, 
            bias=bias
        )
    
    def forward(self, x):
        x = self.norm(x)
        x = self.inception(x)
        return x
    
## Gated-Dconv Feed-Forward Network (GDFN)
class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor = 2.66, 
                 bias = False,
                 LayerNorm_type = "WithBias",
                 act_fn_name = "gelu"
    ):
        super(Gated_Dconv_FeedForward, self).__init__()
        self.norm = LayerNorm(dim, LayerNorm_type = LayerNorm_type)

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.act_fn = ACT_FN[act_fn_name]

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x

def FFN_FN(
    ffn_name,
    dim, 
    ffn_expansion_factor=2.66, 
    bias=False,
    LayerNorm_type="WithBias",
    act_fn_name = "gelu"
):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
                dim, 
                ffn_expansion_factor=ffn_expansion_factor, 
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                act_fn_name = act_fn_name
            )
    

class MixS2Block(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
    ):
        super().__init__()
        self.opt = opt
        dw_channel = dim * opt.MODEL.SR.DW_EXPAND
        if opt.MODEL.SR.SPATIAL_BRANCH:
            self.spatial_branch = SpatialBranch(
                dim, 
                DW_Expand=opt.MODEL.SR.DW_EXPAND,
                bias=opt.MODEL.SR.BIAS,
                LayerNorm_type=opt.MODEL.SR.LAYERNORM_TYPE,
            )
            self.spatial_gelu = nn.GELU()
            self.spatial_conv = nn.Conv2d(in_channels=dw_channel, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=opt.MODEL.SR.BIAS)

        if opt.MODEL.SR.SPATIAL_INTERACTION:
            self.spatial_interaction = nn.Conv2d(dw_channel, 1, kernel_size=1, bias=opt.MODEL.SR.BIAS)

        if opt.MODEL.SR.SPECTRAL_BRANCH:
            self.spectral_branch = SpectralBranch(
                opt,
                dim, 
                num_heads=num_heads, 
                bias=opt.MODEL.SR.BIAS,
                LayerNorm_type=opt.MODEL.SR.LAYERNORM_TYPE
            )

        if opt.MODEL.SR.SPECTRAL_INTERACTION:
            self.spectral_interaction = nn.Sequential(
                nn.Conv2d(dim, dim // 8, kernel_size=1, bias=opt.MODEL.SR.BIAS),
                LayerNorm(dim // 8, opt.MODEL.SR.LAYERNORM_TYPE),
                nn.GELU(),
                nn.Conv2d(dim // 8, dw_channel, kernel_size=1, bias=opt.MODEL.SR.BIAS),
            )

        self.ffn = Residual(
            FFN_FN(
                dim=dim, 
                ffn_name=opt.MODEL.SR.FFN_NAME,
                ffn_expansion_factor=opt.MODEL.SR.FFN_EXPAND, 
                bias=opt.MODEL.SR.BIAS,
                LayerNorm_type=opt.MODEL.SR.LAYERNORM_TYPE
            )
        )

    def forward(self, x):
        log_dict = defaultdict(list)
        b, c, h, w = x.shape

        spatial_fea = 0
        spectral_fea = 0
    
        if self.opt.MODEL.SR.SPATIAL_BRANCH: 
            spatial_identity = x
            spatial_fea = self.spatial_branch(x)
            

        spatial_interaction = None
        if self.opt.MODEL.SR.SPATIAL_INTERACTION:
            spatial_interaction = self.spatial_interaction(spatial_fea)
            log_dict['block_spatial_interaction'] = spatial_interaction
        
        if self.opt.MODEL.SR.SPATIAL_BRANCH:
            spatial_fea = self.spatial_gelu(spatial_fea)

        if self.opt.MODEL.SR.SPECTRAL_BRANCH:
            spectral_identity = x
            spectral_fea = self.spectral_branch(x, spatial_interaction)
        if self.opt.MODEL.SR.SPECTRAL_INTERACTION:
            spectral_interaction = self.spectral_interaction(
                F.adaptive_avg_pool2d(spectral_fea, output_size=1))
            spectral_interaction = torch.sigmoid(spectral_interaction).tile((1, 1, h, w))
            spatial_fea = spectral_interaction * spatial_fea
        if self.opt.MODEL.SR.SPATIAL_BRANCH:
            spatial_fea = self.spatial_conv(spatial_fea)

        if self.opt.MODEL.SR.SPATIAL_BRANCH:
            spatial_fea = spatial_identity + spatial_fea
            log_dict['block_spatial_fea'] = spatial_fea
        if self.opt.MODEL.SR.SPECTRAL_BRANCH:
            spectral_fea = spectral_identity + spectral_fea
            log_dict['block_spectral_fea'] = spectral_fea
        

        fea = spatial_fea + spectral_fea


        out = self.ffn(fea)


        return out
    

class MixS2SR(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.embedding = nn.Conv2d(
            in_channels=opt.MODEL.SR.IN_DIM, 
            out_channels=opt.MODEL.SR.DIM, 
            kernel_size=3, 
            padding=1, 
            stride=1, 
            groups=1,
            bias=True
        )

        self.body = nn.Sequential(
            *[
                Residual(MixS2Block(opt, opt.MODEL.SR.DIM, opt.MODEL.SR.NUM_HEADS))
                for i in range(opt.MODEL.SR.NUM_BLOCKS)
            ]   
        )

        self.up = nn.Sequential(
            nn.Conv2d(
                in_channels=opt.MODEL.SR.DIM, 
                out_channels=opt.MODEL.SR.OUT_DIM * opt.MODEL.SR.UP_SCALE**2, kernel_size=3, 
                padding=1, 
                stride=1, 
                groups=1, 
                bias=True
            ),
            nn.PixelShuffle(opt.MODEL.SR.UP_SCALE)
        )

        self.up_scale = opt.MODEL.SR.UP_SCALE

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        feat = self.embedding(inp)
        feat = self.body(feat)
        out = self.up(feat)
        out = out + inp_hr
        return out
    

    

class Discriminator(nn.Module):
    def __init__(self, inp_nc):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inp_nc, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
    

if __name__ == '__main__':
    from box import Box
    opt = Box(dict)
    model = MixS2SR(opt)
    print(model)
    inp = torch.randn(1, 3, 500, 750)
    out = model(inp)
    print(out.shape)


