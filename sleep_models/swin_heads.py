from msilib.schema import Patch
from tkinter import Variable
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformerBlock, BasicLayer, PatchMerging, PatchEmbed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class PatchMerging_6x1(nn.Module):
    def __init__(self, input_len, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_len = input_len
        self.dim = dim
        self.norm = norm_layer(6 * dim)
        self.reduction = nn.Linear(6 * dim, dim, bias=False)
        
    def forward(self, x):
        """
        x: B, L, C
        """
        L = self.input_len
        B, Lx, C = x.shape
        assert L == Lx, "input feature has wrong size"
        assert L % 6 == 0, f"x size ({L}) is not a multiple of 6."
        x0 = x[:, 0::6, :]  # B L/6 C
        x1 = x[:, 1::6, :]  # B L/6 C
        x2 = x[:, 2::6, :]  # B L/6 C
        x3 = x[:, 3::6, :]  # B L/6 C
        x4 = x[:, 4::6, :]  # B L/6 C
        x5 = x[:, 5::6, :]  # B L/6 C
        x = torch.cat([x0, x1, x2, x3, x4, x5], -1)  # B L/6 6*C
        x = x.view(B, -1, 6 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x

class GAP_head(nn.Module):
    def __init__(self, dim = 48 ,kernel = 8) -> None:
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AvgPool1d(kernel_size=kernel, stride=kernel)
    def forward(self, x):
        """
        x: B, L, C
        """
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = x.transpose(1,2)
        return x

class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_len, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_len = input_len
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_len=input_len ,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_len=input_len, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchExpand(nn.Module):
    def __init__(self, input_len, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_len = input_len
        self.dim = dim
        self.expand = nn.Linear(dim, 4*dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, L, C
        """
        B, Lx, C = x.shape
        L = self.input_len
        assert Lx == L, "input feature has wrong size"
        x = self.expand(x)
        
        x = x.view(B, -1, C)
        x= self.norm(x)
        return x

class HeadStage150s(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        # here 1 window = 5s
        input_len = cfg.SWIN.WINDOW_SIZE * 6 * 5
        # self.downsample = PatchMerging(input_len=input_len, dim=cfg.SWIN.EMBED_DIM)
        self.downsample = GAP_head(dim = cfg.SWIN.EMBED_DIM, 
                                    kernel = cfg.SWIN.WINDOW_SIZE)
        self.layer = BasicLayer(dim=cfg.SWIN.EMBED_DIM,
                            input_len = input_len // cfg.SWIN.WINDOW_SIZE,
                            depth = 4,
                            num_heads = 4,
                            window_size = 30,   
                            mlp_ratio = 2,
                            qkv_bias = True, qk_scale=None,
                            drop = 0, attn_drop = 0.2, drop_path = .2,
                            norm_layer = nn.LayerNorm, 
                            downsample = None,)
        self.norm = nn.LayerNorm(cfg.SWIN.EMBED_DIM)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg.SWIN.EMBED_DIM, cfg.SWIN.OUT_CHANS) 

    def forward(self, x:torch.Tensor, dummy):
        x = self.downsample(x)
        x = self.layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return self.fc(x), x.detach()

class HeadStagePsg(nn.Module):
    def __init__(self, cfg, ape = True) -> None:
        super().__init__()
        self.ape = ape

        self.patch_embed = PatchEmbed(in_len= 5 * 6 * cfg.PSG_EPOCH, patch_size = 5, in_chans=48+60,
                            embed_dim=cfg.SWIN.EMBED_DIM, norm_layer=nn.LayerNorm)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, cfg.PSG_EPOCH * 6, cfg.SWIN.EMBED_DIM))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.layer_epoch = BasicLayer(dim=cfg.SWIN.EMBED_DIM,
                            input_len = cfg.PSG_EPOCH * 6, #input_len,
                            depth = 2,
                            num_heads = 4,
                            window_size = 6, 
                            mlp_ratio = 1,
                            qkv_bias = True, qk_scale=None,
                            drop = 0.2, attn_drop = 0.2, drop_path = .2,
                            norm_layer = nn.LayerNorm, 
                            downsample = PatchMerging_6x1)
        self.layer_down = BasicLayer(dim=cfg.SWIN.EMBED_DIM,
                            input_len = cfg.PSG_EPOCH, #input_len,
                            depth = 4,
                            num_heads = 4,
                            window_size = 35, 
                            mlp_ratio = 1,
                            qkv_bias = True, qk_scale=None,
                            drop = 0.2, attn_drop = 0.2, drop_path = .2,
                            norm_layer = nn.LayerNorm, 
                            downsample = PatchMerging)
        self.layer_up = BasicLayer_up(dim=cfg.SWIN.EMBED_DIM,
                            input_len = cfg.PSG_EPOCH // 4,
                            depth = 4,
                            num_heads = 4,
                            window_size = 45, # 180 epoch = 1 cycle, window num = 45
                            mlp_ratio = 1,
                            qkv_bias = True, qk_scale=None,
                            drop = 0.2, attn_drop = 0.2, drop_path = .2,
                            norm_layer = nn.LayerNorm, 
                            upsample = PatchExpand,) # output = [batch_scale x EMBED_DIM]
        self.conv_head = nn.Conv1d(cfg.SWIN.EMBED_DIM, 6, kernel_size = 1) # [batch_scale x 6]


    def forward(self, x:torch.Tensor, dummy):   # [bs, 1260, 30, 48+60]
        B, _, _, C = x.shape
        x = x.view([B, -1, C]).swapaxes(1,2)        
        x = self.patch_embed(x)                 # [bs, 1260 * 6, 48]
        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.layer_epoch(x)                   # [bs, 1260, 48]
        x_down = self.layer_down(x)               # [bs, 1260 / 4, 48]
        x_up = self.layer_up(x_down)              # [bs, 1260, 48]
        x = x + x_up

        x = x.transpose(1, 2)                   # [bs, 48, 1260]
        x = self.conv_head(x)                   # [bs, 6, 1260]
        x = x.transpose(1, 2)                   # [bs, 1260, 6]
        return x, None