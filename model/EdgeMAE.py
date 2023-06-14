# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat,reduce
from timm.models.vision_transformer import PatchEmbed, Block
from utils.operator import Sobel
from utils.pos_embed import get_2d_sincos_pos_embed

class EdgeMAE(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1,embed_dim=128, depth=24, num_heads=16,
                 decoder_embed_dim=64, decoder_depth=5, decoder_num_heads=16,mlp_ratio=4.,
                  norm_layer=nn.LayerNorm, norm_pix_loss=False,rec_depth=3,edge_depth=3,patchwise_loss=True):
        super().__init__()
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.H = int(img_size / patch_size)
        self.W = int(img_size / patch_size)
        self.patch_size = patch_size
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.rec_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(rec_depth)])

        self.edge_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(edge_depth)])

        self.rec_norm = norm_layer(decoder_embed_dim)
        self.rec_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.rec_sig = nn.Sigmoid()

        self.edge_norm = norm_layer(decoder_embed_dim)
        self.edge_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        self.edge_sig = nn.Sigmoid()

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.operator = Sobel(requires_grad=False)

        self.patchwise_loss = patchwise_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs


    def random_masking(self, x, mask_ratio):

        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x_rec = x
        x_edge = x

        x_rec = self.rec_norm(x_rec)
        x_rec = self.rec_sig(self.rec_pred(x_rec))

        x_edge = self.edge_norm(x_edge)
        x_edge = self.edge_sig(self.edge_pred(x_edge))

        # remove cls token
        x_edge = x_edge[:, 1:, :]
        x_rec = x_rec[:, 1:, :]

        return x_edge,x_rec

    def structure_loss(self, mask, epoch):
        
        weit = rearrange(mask,'b (h w) -> b h w',h=self.H,w=self.W)
        if epoch > 0.5 * epoch:
            weit = 1 + torch.abs(F.avg_pool2d(weit, kernel_size=3, stride=1, padding=1))
        else:
            weit = 2 - torch.abs(F.avg_pool2d(weit, kernel_size=3, stride=1, padding=1))
        weit = repeat(weit,'b h w -> b 1 (h h2) (w w2)', h2=self.patch_size, w2 = self.patch_size)
        
        return weit

    def edge_loss(self, imgs, pred, mask, weit):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        with torch.no_grad():
            edge_gt = self.operator(imgs)
        target = self.patchify(edge_gt)
        weit = self.patchify(weit)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss * weit

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss,edge_gt
    
    def rec_loss(self, imgs, pred, mask, weit):

        target = self.patchify(imgs)
        weit = self.patchify(weit)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        
        if self.patchwise_loss == True:
            loss = loss * weit

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def unpatchify(self,x):

        p = self.patch_size
        h = w = self.H
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def MAE_visualize(self,img, y, mask):
        p = self.patch_size
        h = w = self.H
        y = self.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2)  
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', img).cpu()

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask
        im_paste = rearrange(im_paste,'b w h c -> b c w h')
        im_masked = rearrange(im_masked,'b w h c -> b c w h')
        y = rearrange(y,'b w h c -> b c w h')

        return y[0],im_masked[0],im_paste[0]

    def forward(self, imgs,mask_ratio=0.75,epoch=1):
        
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        x_edge,x_rec= self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        weit = self.structure_loss(mask,epoch)
        rec_loss = self.rec_loss(imgs, x_rec, mask, weit)
        edge_loss,edge_gt = self.edge_loss(imgs, x_edge, mask, weit)
        
        return rec_loss, edge_loss,edge_gt,x_edge,x_rec, mask
    
    
class MAE_finetune(nn.Module):

    def __init__(self, img_size=256,patch_size=8, in_chans=1, embed_dim=128, depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def patchify(self, imgs):

        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        feature = [x]
        for blk in self.blocks:
            x = blk(x)
            feature.append(x)
        x = self.norm(x)
        feature.append(x)
        return feature

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        return latent