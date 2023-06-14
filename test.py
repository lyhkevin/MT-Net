import torch
from utils.dataloader import *
from model.EdgeMAE import *
from model.MTNet import *
from tqdm import tqdm
from utils.test import *
from options import Test_Options
from model.DSF import *

opt = Test_Options().get_opt()
opt.source_modal = 't1'
opt.target_modal = 't2'
opt.img_save_path = './snapshot/test/'
opt.data_root = './data/test/'
opt.E_path = './weight/'
opt.G_path = './weight/'

os.makedirs(opt.img_save_path,exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
E = MAE_finetune(img_size=opt.img_size,patch_size=opt.mae_patch_size, embed_dim=opt.encoder_dim, depth=opt.depth, num_heads=opt.num_heads, in_chans=1, mlp_ratio=opt.mlp_ratio)
G = MTNet(img_size=opt.img_size, patch_size=opt.patch_size, in_chans=1, num_classes=1, embed_dim=opt.vit_dim, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 2], num_heads=[8, 8, 16, 32],window_size=opt.window_size, mlp_ratio=opt.mlp_ratio, qkv_bias=True, qk_scale=None, drop_rate=0.,
          attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,use_checkpoint=False, final_upsample="expand_first", fine_tune=True)

data_loader = get_loader(batchsize=opt.batch_size, shuffle=False, pin_memory=True,source_modal=opt.source_modal, target_modal=opt.target_modal, img_size=opt.img_size,num_workers=opt.num_workers,
                         img_root=opt.data_root, data_rate=opt.data_rate, argument=False, random=False)

E = E.to(device)
G = G.to(device)    
E.load_state_dict(torch.load(opt.E_path, map_location=torch.device(device)),strict=False)
G.load_state_dict(torch.load(opt.G_path, map_location=torch.device(device)),strict=False)

PSNR = []
NMSE = []
SSIM = []

for i,(img, gt) in enumerate(data_loader):

    batch_size = img.size()[0]
    img = img.to(device,dtype=torch.float)
    gt = gt.to(device,dtype=torch.float)

    with torch.no_grad():
        Feature = E(img)
        f1, f2 = Feature[-1].clone(), Feature[-1].clone()
        pred = G(f1, f2)
        
    for j in range(batch_size):
        save_image([pred[j]], opt.img_save_path + str(i*opt.batch_size + j + 1)+'.png',normalize=False)
        print(opt.img_save_path + str(i*opt.batch_size + j + 1)+'.png')

    pred,gt = pred.cpu().detach().numpy().squeeze(), gt.cpu().detach().numpy().squeeze()

    for j in range(batch_size):
        PSNR.append(psnr(pred[j], gt[j]))
        NMSE.append(nmse(pred[j], gt[j]))
        SSIM.append(ssim(pred[j], gt[j]))

PSNR = np.asarray(PSNR)
NMSE = np.asarray(NMSE)
SSIM = np.asarray(SSIM)

PSNR = PSNR.reshape(-1, opt.slice_num)
NMSE = NMSE.reshape(-1, opt.slice_num)
SSIM = SSIM.reshape(-1, opt.slice_num)

PSNR = np.mean(PSNR,axis=1)
NMSE = np.mean(NMSE,axis=1)
SSIM = np.mean(SSIM,axis=1)

print("PSNR mean:",np.mean(PSNR),"PSNR std:",np.std(PSNR))
print("NMSE mean:",np.mean(NMSE),"NMSE std:",np.std(NMSE))
print("SSIM mean:",np.mean(SSIM),"SSIM std:",np.std(SSIM))