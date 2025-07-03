import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from swin_transformer import PatchEmbed, BasicLayer


"""
Hierarchically-Guided Swin Transformer, HG-Swin
"""


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    """
    def __init__(self, in_chans, out_chans, reduction=16):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.reduction = reduction

        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

        self.excitation = nn.Sequential(
            nn.Linear(in_chans, in_chans // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_chans // reduction, out_chans, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: input tensor of shape (B, IN_C, H, W)
        :return y: output tensor of shape (B, OUT_C)
        """
        B, C, _, _ = x.shape
        y = self.squeeze(x).view(B, C)
        y = self.excitation(y)
        return y

    def extra_expr(self):
        return f"in_channels={self.in_chans}, out_channels={self.out_chans}, reduction={self.reduction}"

    def flops(self, H, W):
        flops = 0
        # squeeze
        flops += self.in_chans * H * W
        # excitation
        flops += self.in_chans * (self.in_chans // self.reduction)
        flops += (self.in_chans // self.reduction) * self.out_chans
        return flops


class AttentivePatchMerging(nn.Module):
    """
    Attentive Patch Merging with Cross-Resolution Squeeze-and-Excitation.
    This layer uses the intermediate 4C-channel tensor (after concatenation) to generate attention weights,
    which are then applied to the final 2C-channel output tensor (after linear reduction).
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

        self.se_block = SqueezeExcitationBlock(in_chans=4 * dim, out_chans=2 * dim)

    def forward(self, x):
        # x: (B, H * W, C)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size."
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H} * {W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # (B, H / 2, W / 2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H / 2, W / 2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H / 2, W / 2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H / 2, W / 2, C)
        x_4c = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H / 2, W / 2, 4C)

        # generate excitation weights from the 4C-channel tensor
        x_se = x_4c.permute(0, 3, 1, 2).contiguous()  # (B, 4C, H / 2, W / 2)
        excitation_weights = self.se_block(x_se)

        x_merged = x_4c.view(B, -1, 4 * C)  # (B, H / 2 * W / 2, 4C)
        x_merged = self.norm(x_merged)
        x_reduced = self.reduction(x_merged)  # (B, H / 2 * W / 2, 2C)

        H_merged, W_merged = H // 2, W // 2

        # apply the excitation weights to the merged features
        x_reduced_reshaped = x_reduced.view(B, 2 * C, H_merged, W_merged)  # (B, 2C, H / 2, W / 2)
        excitation_weights = excitation_weights.view(B, 2 * C, 1, 1)  # (B, 2C, 1, 1)
        x_excited = x_reduced_reshaped * excitation_weights.expand_as(x_reduced_reshaped)  # (B, 2C, H / 2, W / 2)
        x_excited = x_excited.view(B, H_merged * W_merged, 2 * C)  # (B, H / 2 * W / 2, 2C)
        return x_excited  # (B, H / 2 * W / 2, 2C)

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        H_merged, W_merged = H // 2, W // 2
        # patch merging
        flops = H * W * self.dim
        flops += H_merged * W_merged * 4 * self.dim * 2 * self.dim
        # se block
        flops += self.se_block.flops(H_merged, W_merged)
        return flops


class HierarchicallyGuidedSwinTransformer(nn.Module):
    """
    Hierarchically-Guided Swin Transformer.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        """
        :param img_size (int): Input image size. Default: 224
        :param patch_size (int | tuple[int]): Patch size. Default: 4
        :param in_chans (int): Number of input image channels. Default: 3
        :param num_classes (int): Number of classes for classification head. Default: 1000
        :param embed_dim (int): Patch embedding dimension. Default: 96
        :param depths (tuple(int)): Depth of each Swin Transformer layer.
        :param num_heads (tuple(int)): Number of attention heads in different layers.
        :param window_size (int): Window size. Default: 7
        :param mlp_ratio (float): Ratio of mlp hidden dimension to embedding dimension. Default: 4
        :param qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        :param drop_rate (float): Dropout rate. Default: 0
        :param attn_drop_rate (float): Attention dropout rate. Default: 0
        :param drop_path_rate (float): Stochastic depth rate. Default: 0.1
        :param norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        :param ape (bool): If True, add absolute positional embedding to the patch embedding. Default: False
        :param patch_norm (bool): If True, add normalization after patch embedding. Default: True
        :param use_checkpoint (bool): Whether to use checkpointing to save money. Default: False
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute positional embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.2)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=AttentivePatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # (B, L, C)
        x = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def build_hierarchically_guided_swin_transformer(args):
    """
    Build a Hierarchically-Guided Swin Transformer model based on the provided arguments.

    :param args: Arguments containing model configuration.
    :return: CRSEST model.
    """
    crsest_parameters = {
        'HG-Swin-T': {'embed_dim': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24]},
        'HG-Swin-S': {'embed_dim': 96, 'depths': [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24]},
        'HG-Swin-B': {'embed_dim': 128, 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32]},
        'HG-Swin-L': {'embed_dim': 192, 'depths': [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48]},
    }

    config = crsest_parameters.get(args.model_name)

    if config is None:
        valid_models = list(crsest_parameters.keys())
        raise ValueError(
            f"Invalid model name '{args.model_name}'. "
            f"Please choose from: {valid_models}."
        )

    model = HierarchicallyGuidedSwinTransformer(
        img_size=getattr(args, 'img_size', 224),
        patch_size=getattr(args, 'patch_size', 4),
        in_chans=getattr(args, 'in_chans', 3),
        num_classes=getattr(args, 'num_classes', 1000),
        embed_dim=config['embed_dim'],
        depths=config['depths'],
        num_heads=config['num_heads'],
        window_size=getattr(args, 'window_size', 7),
        mlp_ratio=getattr(args, 'mlp_ratio', 4.),
        qkv_bias=getattr(args, 'qkv_bias', True),
        qk_scale=getattr(args, 'qk_scale', None),
        drop_rate=getattr(args, 'drop_rate', 0.),
        attn_drop_rate=getattr(args, 'attn_drop_rate', 0.),
        drop_path_rate=getattr(args, 'drop_path_rate', 0.1),
        norm_layer=nn.LayerNorm,
        ape=getattr(args, 'ape', False),
        patch_norm=getattr(args, 'patch_norm', True),
        use_checkpoint=getattr(args, 'use_checkpoint', False)
    )

    return model


if __name__ == "__main__":
    import argparse
    # Example usage
    args = {
        'model_name': 'HG-Swin-T',
        'img_size': 224,
        'patch_size': 4,
        'in_chans': 3,
        'num_classes': 3,
        'window_size': 7,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False
    }
    args = argparse.Namespace(**args)

    model = build_hierarchically_guided_swin_transformer(args)
    print(model)
    print(model.flops())

    x = torch.randn(1, args.in_chans, args.img_size, args.img_size)
    print(f"Model Output: {model(x)}")
    print(f"Model Output shape: {model(x).shape}")
