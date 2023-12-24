from functools import partial
from typing import Callable, Optional
import numpy as np
import mindspore as ms
import mindspore
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import Normal, initializer
from mindspore import load_checkpoint, load_param_into_net

from model.mindcv.models.helpers import load_pretrained
from model.mindcv.models.layers.mlp import Mlp
from model.mindcv.models.layers.patch_embed import PatchEmbed
from model.mindcv.models.registry import register_model
from model.mindcv.models.vit import Block, VisionTransformer

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class EdgeMAE_finetune(nn.Cell):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        init_values: Optional[float] = None,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        mlp_layer: Callable = Mlp,
        norm_pix_loss: bool = False,
        mask_ratio: float = 0.75,
        FC_module = False,
        **kwargs,
    ):
        super(EdgeMAE_finetune, self).__init__()
        self.patch_embed = PatchEmbed(image_size=image_size, patch_size=patch_size,
                                      in_chans=in_channels, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.CellList([
            Block(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                attn_drop=attn_drop_rate, proj_drop=proj_drop_rate,
                mlp_ratio=mlp_ratio, drop_path=dpr[i], init_values=init_values,
                act_layer=act_layer, norm_layer=norm_layer, mlp_layer=mlp_layer,
            ) for i in range(depth)
        ])

        self.cls_token = Parameter(initializer(Normal(sigma=0.02), (1, 1, embed_dim)))

        self.unmask_len = int(np.floor(self.num_patches * (1 - mask_ratio)))

        encoder_pos_emb = Tensor(get_2d_sincos_pos_embed(
            embed_dim, int(self.num_patches ** 0.5), cls_token=True), ms.float32
        )
        encoder_pos_emb = ops.expand_dims(encoder_pos_emb, axis=0)
        self.pos_embed = Parameter(encoder_pos_emb, requires_grad=False)
        self.norm = norm_layer((embed_dim,))

        self.sort = ops.Sort()
        self.norm_pix_loss = norm_pix_loss
        self.FC_module = FC_module
        self._init_weights()

    def _init_weights(self):
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    initializer("xavier_uniform", cell.weight.shape, cell.weight.dtype)
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
            if name == "patch_embed.proj":
                cell.weight.set_data(
                    initializer("xavier_uniform", cell.weight.shape, cell.weight.dtype)
                )

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size ** 2 * 3)
        """
        N, _, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        assert H == W and H % p == 0
        h = w = H // p
        x = ops.reshape(imgs, (N, 1, h, p, w, p))
        x = ops.transpose(x, (0, 2, 4, 3, 5, 1))
        x = ops.reshape(x, (N, h * w, p ** 2 * 1))
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

        imgs = ops.reshape(x, (N, h, w, p, p, 3))
        imgs = ops.transpose(imgs, (0, 5, 1, 3, 2, 4))
        imgs = ops.reshape(imgs, (N, 3, h * p, w * p))
        return imgs

    def forward_features(self, x):
        x = self.patch_embed(x)
        bsz = x.shape[0]

        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = ops.broadcast_to(cls_token, (bsz, -1, -1))
        cls_token = cls_token.astype(x.dtype)
        x = ops.concat((cls_token, x), axis=1)

        if self.FC_module == True:
            features = []
            for blk in self.blocks:
                x = blk(x)
                features.append(x)
            return features
        else:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x[:, 1:, :]

    def construct(self, imgs):
        features = self.forward_features(imgs)
        return features

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


class Generator(nn.Cell):
    def __init__(self, EdgeMAE, TransUNet):
        super(Generator, self).__init__()
        self.EdgeMAE = EdgeMAE
        self.TransUNet = TransUNet

    def construct(self, imgs):
        features = self.EdgeMAE(imgs)
        pred = self.TransUNet(features)
        return pred

def get_mask(size = 784, mask_rate = 0.7, batch_size=2):
    mask = np.zeros((batch_size, size))
    mask[:, :int(size * mask_rate)] = 1
    np.apply_along_axis(np.random.shuffle, axis=1, arr=mask)
    mask = Tensor(mask, mindspore.float32)
    return mask

if __name__ == '__main__':
    image_size, patch_size = 256, 8
    num_patches = (image_size // patch_size) ** 2
    batch_size = 10
    mask = get_mask(size=num_patches, mask_rate=0.75, batch_size=batch_size)
    model = EdgeMAE_finetune(image_size=image_size, patch_size=patch_size, in_channels=1, embed_dim=384, depth=12, num_heads=12,
                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,act_layer=partial(nn.GELU, approximate=False),
                    norm_layer=partial(nn.LayerNorm, epsilon=1e-6))
    param_dict = load_checkpoint("../../weight/EdgeMAE/model.ckpt")
    load_param_into_net(model, param_dict)
    print('successfully load params')
    image_tensor = Tensor(np.random.randn(batch_size, 1, 256, 256), ms.float32)
    features = model(imgs = image_tensor)
    print(features.shape)