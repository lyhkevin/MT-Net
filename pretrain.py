import torch
import logging
from utils.maeloader import *
from model.EdgeMAE import *
from utils.mae_visualize import *
from options import Pretrain_Options

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
opt = Pretrain_Options().get_opt()

mae = EdgeMAE(img_size=opt.img_size,patch_size=opt.patch_size, embed_dim=opt.dim_encoder, depth=opt.depth, num_heads=opt.num_heads, in_chans=1,
        decoder_embed_dim=opt.dim_decoder, decoder_depth=opt.decoder_depth, decoder_num_heads=opt.decoder_num_heads,
        mlp_ratio=opt.mlp_ratio,norm_pix_loss=False,patchwise_loss=opt.use_patchwise_loss)

os.makedirs(opt.img_save_path,exist_ok=True)
os.makedirs(opt.weight_save_path,exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader = get_maeloader(batchsize=opt.batch_size, shuffle=True,pin_memory=True,img_size=opt.img_size,
            img_root=opt.data_root,num_workers=opt.num_workers,augment=opt.augment,modality=opt.modality)

optimizer = torch.optim.Adam(mae.parameters(), lr=opt.lr,betas=(0.9, 0.95))
mae = mae.to(device)

if opt.use_checkpoints == True:
    print('load checkpoint......',opt.checkpoint_path)
    mae.load_state_dict(torch.load(opt.checkpoint_path, map_location=torch.device(device)),strict=False)
    
logging.basicConfig(filename=opt.log_path,
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

for epoch in range(1,opt.epoch):
    for i,img in enumerate(train_loader):

        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        
        optimizer.zero_grad()

        img = img.to(device,dtype=torch.float)

        rec_loss, edge_loss,edge_gt,x_edge,x_rec,mask = mae(img,opt.masking_ratio,epoch)
        loss = rec_loss * opt.l1_loss + edge_loss 
        
        loss.backward()
        optimizer.step()

        print(
                "[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]"
                % (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer))
            )

        if i % opt.save_output == 0:
            y1, im_masked1, im_paste1 = mae.MAE_visualize(img, x_rec, mask)
            y2, im_masked2, im_paste2 = mae.MAE_visualize(edge_gt, x_edge, mask)
            edge_gt,img = edge_gt.cpu(),img.cpu()
            save_image([img[0],im_masked1,im_paste1,edge_gt[0],im_masked2,im_paste2],
                 opt.img_save_path + str(epoch) + ' ' + str(i)+'.png', nrow=3,normalize=False)
            logging.info("[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]"
                % (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer)))

    if epoch % opt.save_weight == 0:
        torch.save(mae.state_dict(), opt.weight_save_path + str(epoch) + 'MAE.pth')

torch.save(mae.state_dict(), opt.weight_save_path + './MAE.pth')
