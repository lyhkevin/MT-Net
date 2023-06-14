import torch
import argparse
from tqdm import tqdm
from utils.dataloader import *
import logging
from utils.test import *
from model.EdgeMAE import *
from model.MTNet import *
from options import Finetune_Options
import torch.nn.functional as F

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

opt = Finetune_Options().get_opt()
os.makedirs(opt.img_save_path, exist_ok=True)
os.makedirs(opt.weight_save_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

E = MAE_finetune(img_size=opt.img_size, patch_size=opt.mae_patch_size, embed_dim=opt.encoder_dim, depth=opt.depth,
                 num_heads=opt.num_heads, in_chans=1, mlp_ratio=opt.mlp_ratio)
FC_module = MAE_finetune(img_size=opt.img_size, patch_size=opt.mae_patch_size, embed_dim=opt.encoder_dim,
                         depth=opt.fc_depth, num_heads=opt.num_heads, in_chans=1,
                         mlp_ratio=opt.mlp_ratio)  # feature consistency module

G = MTNet(img_size=opt.img_size, patch_size=opt.patch_size, in_chans=1, num_classes=1, embed_dim=opt.vit_dim,
          depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 2], num_heads=[8, 8, 16, 32], window_size=opt.window_size,
          mlp_ratio=opt.mlp_ratio, qkv_bias=True, qk_scale=None, drop_rate=0.,
          attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
          use_checkpoint=False, final_upsample="expand_first", fine_tune=True)

data_loader = get_loader(batchsize=opt.batch_size, shuffle=True, pin_memory=True, source_modal=opt.source_modal,
                         target_modal=opt.target_modal, img_size=opt.img_size, num_workers=opt.num_workers,
                         img_root=opt.data_root, data_rate=opt.data_rate, argument=True, random=False)

E.load_state_dict(torch.load(opt.mae_path, map_location=torch.device(device)), strict=False)
FC_module.load_state_dict(torch.load(opt.mae_path, map_location=torch.device(device)), strict=False)
lr_schedule = cosine_scheduler(opt.lr, opt.min_lr, opt.epoch, len(data_loader), warmup_epochs=opt.warmup_epochs)

E = E.to(device)
G = G.to(device)
FC_module = FC_module.to(device)

for param in FC_module.parameters():
    param.requires_grad = False

params = list(E.parameters()) + list(G.parameters())
optimizer = torch.optim.Adam(params)

logging.basicConfig(filename=opt.log_path,
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

for epoch in range(1, opt.epoch):
    for i, (img, gt) in enumerate(data_loader):
        it = len(data_loader) * epoch + i
        for id, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]

        optimizer.zero_grad()
        img = img.to(device, dtype=torch.float)
        gt = gt.to(device, dtype=torch.float)

        Feature = E(img)
        f1, f2 = Feature[-1].clone(), Feature[-1].clone()
        pred = G(f1, f2)
        feature = FC_module(pred)
        feature_gt = FC_module(gt.detach())

        l1_loss = F.l1_loss(pred, gt)
        fc_loss = 0  # feature consistency loss
        for j in range(opt.fc_depth):
            fc_loss = fc_loss + F.l1_loss(feature[j], feature_gt[j])
        loss = opt.l1_loss * l1_loss + fc_loss
        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [L1 loss: %f] [feat_loss: %f] [lr: %f]"
            % (epoch, opt.epoch, i, len(data_loader), l1_loss.item(), fc_loss.item(), get_lr(optimizer))
        )

        if i % opt.save_snapshot == 0:
            save_image([img[0], gt[0], pred[0]], opt.img_save_path + str(epoch) + ' ' + str(i) + '.png', normalize=True)
            logging.info("[Epoch %d/%d] [Batch %d/%d] [L1 loss: %f] [feat_loss: %f] [lr: %f]"
                         % (epoch, opt.epoch, i, len(data_loader), l1_loss.item(), fc_loss.item(), get_lr(optimizer)))

    if epoch % opt.save_weight == 0:
        torch.save(E.state_dict(), opt.weight_save_path + str(epoch) + 'E.pth')
        torch.save(G.state_dict(), opt.weight_save_path + str(epoch) + 'G.pth')

torch.save(E.state_dict(), opt.weight_save_path + 'E.pth')
torch.save(G.state_dict(), opt.weight_save_path + 'G.pth')