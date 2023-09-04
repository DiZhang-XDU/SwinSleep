from .swin_transformer import SwinTransformer, PatchMerging
from torch import nn


class Build_Swin(nn.Module):
    def __init__(self, body, head) -> None:
        super().__init__()
        self.body = body
        self.head = head
    def forward(self, x):
        x, x_downsample = self.body(x)
        x, deep_feature = self.head(x, x_downsample)
        return x, deep_feature

class _body_Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x, x

def build_model(cfg):
    if cfg.net == 'swin':
        body = SwinTransformer(image_size=cfg.SWIN.IN_LEN,
                                    patch_size=cfg.SWIN.PATCH_SIZE,
                                    in_chans=cfg.SWIN.IN_CHANS,
                                    num_classes=cfg.SWIN.OUT_CHANS,
                                    embed_dim=cfg.SWIN.EMBED_DIM,
                                    depths=cfg.SWIN.DEPTHS,
                                    num_heads=cfg.SWIN.NUM_HEADS,
                                    window_size=cfg.SWIN.WINDOW_SIZE,
                                    mlp_ratio=cfg.SWIN.MLP_RATIO,
                                    qkv_bias=cfg.SWIN.QKV_BIAS,
                                    qk_scale=cfg.SWIN.QK_SCALE,
                                    drop_rate=cfg.SWIN.DROP_RATE,
                                    drop_path_rate=cfg.SWIN.DROP_PATH_RATE,
                                    ape=cfg.SWIN.APE,
                                    patch_norm=cfg.SWIN.PATCH_NORM,
                                    head=cfg.HEAD,
                                    use_checkpoint=cfg.USE_CHECKPOINT)

        if cfg.HEAD == 'stage150':
            from .swin_heads import HeadStage150s
            head = HeadStage150s(cfg) 
        elif cfg.HEAD == 'stagepsg':
            body = _body_Identity() ############
            from .swin_heads import HeadStagePsg
            head = HeadStagePsg(cfg)
        else:
            raise NotImplementedError(f"Unkown model: {cfg.head}")
        
        model = Build_Swin(body, head)

    return model
