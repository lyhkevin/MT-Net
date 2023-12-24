"""ViT"""
from typing import Callable, List, Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import HeUniform, TruncatedNormal, initializer

from model.mindcv.models.helpers import load_pretrained
from model.mindcv.models.layers.compatibility import Dropout
from model.mindcv.models.layers.drop_path import DropPath
from model.mindcv.models.layers.mlp import Mlp
from model.mindcv.models.layers.patch_dropout import PatchDropout
from model.mindcv.models.layers.patch_embed import PatchEmbed
from model.mindcv.models.layers.pos_embed import resample_abs_pos_embed
from model.mindcv.models.registry import register_model

# TODO: Flash Attention
class Attention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.

    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        qkv_bias (bool): Specifies whether the linear layer uses a bias vector. Default: True.
        qk_norm (bool): Specifies whether to do normalization to q and k.
        attn_drop (float): The drop rate of attention, greater than 0 and less equal than 1. Default: 0.0.
        proj_drop (float): The drop rate of output, greater than 0 and less equal than 1. Default: 0.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = Attention(768, 12)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Cell = nn.LayerNorm,
    ):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = Tensor(self.head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.q_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()
        self.k_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()

        self.attn_drop = Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = Dropout(proj_drop)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def construct(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, self.head_dim))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)

        attn = attn.astype(ms.float32)
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class LayerScale(nn.Cell):
    """
    Layer scale, help ViT improve the training dynamic, allowing for the training
    of deeper high-capacity image transformers that benefit from depth

    Args:
        dim (int): The output dimension of attnetion layer or mlp layer.
        init_values (float): The scale factor. Default: 1e-5.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = LayerScale(768, 0.01)
    """
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5
    ):
        super(LayerScale, self).__init__()
        self.gamma = Parameter(initializer(init_values, dim))

    def construct(self, x):
        return self.gamma * x


class Block(nn.Cell):
    """
    Transformer block implementation.

    Args:
        dim (int): The dimension of embedding.
        num_heads (int): The number of attention heads.
        qkv_bias (bool): Specifies whether the linear layer uses a bias vector. Default: True.
        attn_drop (float): The drop rate of attention, greater than 0 and less equal than 1. Default: 0.0.
        proj_drop (float): The drop rate of dense layer output, greater than 0 and less equal than 1. Default: 0.0.
        mlp_ratio (float): The ratio used to scale the input dimensions to obtain the dimensions of the hidden layer.
        drop_path (float): The drop rate for drop path. Default: 0.0.
        act_layer (nn.Cell): Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm_layer (nn.Cell): Norm layer that will be stacked on top of the convolution
            layer. Default: nn.LayerNorm.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = TransformerEncoder(768, 12, 12, 3072)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        mlp_layer: Callable = Mlp,
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop
        )
        self.ls2 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class PatchMerging(nn.Cell):
    def __init__(
        self,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: Optional[nn.Cell] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim[0] if isinstance(dim, tuple) and len(dim) == 1 else dim
        # Default False
        self.reduction = nn.Dense(in_channels=4 * dim, out_channels=dim, has_bias=False)
        self.norm = norm_layer([dim * 4, ])
        self.H, self.W = self.input_resolution
        self.H_2, self.W_2 = self.H // 2, self.W // 2
        self.H2W2 = int(self.H * self.W // 4)
        self.dim_mul_4 = int(dim * 4)
        self.H2W2 = int(self.H * self.W // 4)

    def construct(self, x: Tensor) -> Tensor:
        """
        x: B, H*W, C
        """
        b = x.shape[0]
        x = ops.reshape(x, (b, self.H_2, 2, self.W_2, 2, self.dim))
        x = ops.transpose(x, (0, 1, 3, 4, 2, 5))
        x = ops.reshape(x, (b, self.H2W2, self.dim_mul_4))
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

class PatchUpsample(nn.Cell):
    def __init__(
        self,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: Optional[nn.Cell] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.upsample = nn.Conv2dTranspose(in_channels=dim, out_channels=dim, kernel_size=2, stride=2)
        self.dim = dim[0] if isinstance(dim, tuple) and len(dim) == 1 else dim
        self.reduction = nn.Dense(in_channels=2 * dim, out_channels=dim, has_bias=False)
        self.norm = norm_layer([dim, ])

    def construct(self, x_small, x_large):
        """
        x: B, H*W, C
        """
        b = x_small.shape[0]
        patch_num = x_large.shape[1]
        x_small = ops.reshape(x_small, (b, self.dim, self.input_resolution[0], self.input_resolution[1]))
        x_small = self.upsample(x_small)
        x_small = ops.transpose(x_small, (0, 2, 3, 1))
        x_small = ops.reshape(x_small, (b, patch_num, self.dim))
        x = ops.concat((x_small, x_large), axis=2)
        x = self.reduction(x)
        x = self.norm(x)
        return x

class TransUNet(nn.Cell):
    '''
    ViT encoder, which returns the feature encoded by transformer encoder.
    '''
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        global_pool: str = 'token',
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        weight_init: bool = True,
        init_values: Optional[float] = None,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        act_layer: nn.Cell = nn.GELU,
        embed_layer: Callable = PatchEmbed,
        norm_layer: nn.Cell = nn.LayerNorm,
        mlp_layer: Callable = Mlp,
        class_token: bool = True,
        block_fn: Callable = Block,
        num_classes: int = 1000,
    ):
        super(TransUNet, self).__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm

        self.global_pool = global_pool
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        elif dynamic_img_pad:
            embed_args.update(dict(output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(initializer(TruncatedNormal(0.02), (1, 1, embed_dim))) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = Parameter(initializer(TruncatedNormal(0.02), (1, embed_len, embed_dim)))
        self.pos_drop = Dropout(pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()

        self.norm_pre = norm_layer((embed_dim,)) if pre_norm else nn.Identity()
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, 4)]

        self.encoder_blocks = nn.CellList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                attn_drop=attn_drop_rate, proj_drop=proj_drop_rate,
                mlp_ratio=mlp_ratio, drop_path=dpr[i], init_values=init_values,
                act_layer=act_layer, norm_layer=norm_layer, mlp_layer=mlp_layer,
            ) for i in range(4)
        ])

        self.decoder_blocks = nn.CellList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                attn_drop=attn_drop_rate, proj_drop=proj_drop_rate,
                mlp_ratio=mlp_ratio, drop_path=dpr[i], init_values=init_values,
                act_layer=act_layer, norm_layer=norm_layer, mlp_layer=mlp_layer,
            ) for i in range(4)
        ])

        self.downsample = nn.CellList([
            PatchMerging(input_resolution = (32,32), dim=embed_dim, norm_layer=nn.LayerNorm),
            PatchMerging(input_resolution = (16,16), dim=embed_dim, norm_layer=nn.LayerNorm),
            PatchMerging(input_resolution = (8,8), dim=embed_dim, norm_layer=nn.LayerNorm),
            PatchMerging(input_resolution = (4,4), dim=embed_dim, norm_layer=nn.LayerNorm),
        ])

        self.upsample = nn.CellList([
            PatchUpsample(input_resolution=(2, 2), dim=embed_dim, norm_layer=nn.LayerNorm),
            PatchUpsample(input_resolution=(4, 4), dim=embed_dim, norm_layer=nn.LayerNorm),
            PatchUpsample(input_resolution=(8, 8), dim=embed_dim, norm_layer=nn.LayerNorm),
            PatchUpsample(input_resolution=(16, 16), dim=embed_dim, norm_layer=nn.LayerNorm),
        ])

        self.norm = norm_layer((embed_dim,)) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer((embed_dim,)) if use_fc_norm else nn.Identity()
        self.head_drop = Dropout(drop_rate)
        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.decoder_pred = nn.Dense(embed_dim, patch_size ** 2 * in_channels)

        if weight_init:
            self._init_weights()

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    initializer(TruncatedNormal(0.02), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        initializer('zeros', cell.bias.shape, cell.bias.dtype)
                    )
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(
                    initializer('ones', cell.gamma.shape, cell.gamma.dtype)
                )
                cell.beta.set_data(
                    initializer('zeros', cell.beta.shape, cell.beta.dtype)
                )
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    initializer(HeUniform(), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        initializer("zeros", cell.bias.shape, cell.bias.dtype)
                    )

    def _pos_embed(self, x):
        if self.dynamic_img_size or self.dynamic_img_pad:
            # bhwc format
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = ops.reshape(x, (B, -1, C))
        else:
            pos_embed = self.pos_embed

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if self.cls_token is not None:
                cls_tokens = ops.broadcast_to(self.cls_token, (x.shape[0], -1, -1))
                cls_tokens = cls_tokens.astype(x.dtype)
                x = ops.concat((cls_tokens, x), axis=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                cls_tokens = ops.broadcast_to(self.cls_token, (x.shape[0], -1, -1))
                cls_tokens = cls_tokens.astype(x.dtype)
                x = ops.concat((cls_tokens, x), axis=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x1 = self.downsample[0](self.encoder_blocks[0](x))
        x2 = self.downsample[1](self.encoder_blocks[0](x1))
        x3 = self.downsample[2](self.encoder_blocks[0](x2))
        x4 = self.downsample[3](self.encoder_blocks[0](x3))
        x5 = self.decoder_blocks[0](self.upsample[0](x4, x3))
        x5 = self.decoder_blocks[1](self.upsample[1](x5, x2))
        x5 = self.decoder_blocks[2](self.upsample[2](x5, x1))
        x = self.decoder_blocks[3](self.upsample[3](x5, x))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size ** 2 * 3)
        imgs: (N, 3, H, W)
        """
        N, L, _ = x.shape
        p = self.patch_embed.patch_size[0]
        h = w = int(L ** 0.5)
        assert h * w == L

        imgs = ops.reshape(x, (N, h, w, p, p, 1))
        imgs = ops.transpose(imgs, (0, 5, 1, 3, 2, 4))
        imgs = ops.reshape(imgs, (N, 1, h * p, w * p))
        return imgs

    def construct(self, x):
        x = self.forward_features(x)
        x = self.decoder_pred(x)
        x = self.unpatchify(x)
        return x

if __name__ == '__main__':
    model = TransUNet(patch_size=8, in_channels=1, embed_dim=384, num_heads=16, num_classes=1)
    img = Tensor(np.random.randn(1, 1024, 368), ms.float32)
    output = model(img)
    print(output.shape)