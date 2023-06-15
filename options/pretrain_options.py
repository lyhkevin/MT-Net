import argparse
import os

class Pretrain_Options():
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
        self.parser.add_argument("--batch_size", default=20, type=int)
        self.parser.add_argument("--epoch", default=100, type=int)
        self.parser.add_argument("--patch_size", default=8, type=int)
        self.parser.add_argument("--img_size", default=256, type=int)
        self.parser.add_argument("--decay_epoch", default=50, type=float)
        self.parser.add_argument("--decay_rate", default=0.1, type=float)
        self.parser.add_argument("--l1_loss", default=10, type=int)
        self.parser.add_argument("--augment", default=True) #preform data augmentation
        self.parser.add_argument("--modality", default='all') #using all modalities for pre-training (t1, t2, t1c, flair)
        self.parser.add_argument("--masking_ratio", default=0.7,type=float)
        self.parser.add_argument("--num_workers", default=0, type=int)
        
        self.parser.add_argument('--use_checkpoints', default=False)
        self.parser.add_argument('--img_save_path', type=str,default='./snapshot/EdgeMAE/')
        self.parser.add_argument('--weight_save_path', type=str,default='./weight/EdgeMAE/')
        self.parser.add_argument('--checkpoint_path', type=str,default='./weight/EdgeMAE/EdgeMAE.pth')
        self.parser.add_argument("--data_root", default='./data/train/')

        self.parser.add_argument("--depth", default=12, type=int)
        self.parser.add_argument("--use_patchwise_loss", default=True)
        self.parser.add_argument("--decoder_depth", default=8, type=int)
        self.parser.add_argument("--save_output", default=200, type=int)
        self.parser.add_argument("--save_weight", default=10, type=int)
        self.parser.add_argument("--num_heads", default=16, type=int)
        self.parser.add_argument("--decoder_num_heads", default=8, type=int)
        self.parser.add_argument("--mlp_ratio", default=4, type=int)
        self.parser.add_argument("--dim_encoder", default=128, type=int)
        self.parser.add_argument("--dim_decoder", default=64, type=int)
        self.parser.add_argument("--log_path", default='./log/EdgeMAE.log')

    def get_opt(self):
        self.opt = self.parser.parse_args()
        return self.opt

        

