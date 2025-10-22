from torchvision.models.swin_transformer import _swin_transformer, SwinTransformerBlockV2, PatchMergingV2


def swin_v2_tv(**kwargs):
    return _swin_transformer(
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        weights=None,
        progress=False,
        **kwargs
    )
