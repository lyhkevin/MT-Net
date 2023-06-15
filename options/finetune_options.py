import argparse
import os


class Finetune_Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument("--warmup_epochs", default=5)
        self.parser.add_argument("--min_lr", default=1e-6)
        self.parser.add_argument("--epoch", default=150, type=int)
        self.parser.add_argument("--batch_size", default=10, type=int)
        self.parser.add_argument("--source_modal", default='t1')
        self.parser.add_argument("--target_modal", default='t2')
        self.parser.add_argument("--data_rate", default=0.7, type=float)  # training data ratio
        self.parser.add_argument("--mae_path", default='./weight/EdgeMAE/EdgeMAE.pth')
        self.parser.add_argument('--img_save_path', type=str, default='./snapshot/finetune/t1_t2_70%/')
        self.parser.add_argument('--weight_save_path', type=str, default='./weight/t1_t2_70%/')
        self.parser.add_argument("--num_workers", default=0, type=int)
        self.parser.add_argument("--img_size", default=256, type=int)
        self.parser.add_argument("--window_size", default=8, type=int)
        self.parser.add_argument("--mae_patch_size", default=8, type=int)
        self.parser.add_argument("--depth", default=12, type=int)  # pre-trained encoder depth
        self.parser.add_argument("--fc_depth", default=12, type=int)  # depth of the feature consistency module

        self.parser.add_argument("--patch_size", default=4, type=int)
        self.parser.add_argument("--data_root", default='./data/train/')
        self.parser.add_argument("--encoder_dim", default=128, type=int)
        self.parser.add_argument("--decoder_dim", default=64, type=int)
        self.parser.add_argument("--vit_dim", default=128, type=int)
        self.parser.add_argument("--l1_loss", default=10000, type=float) # l1_loss : feature consistency loss = 10000: 1
        self.parser.add_argument("--num_heads", default=16, type=int)
        self.parser.add_argument("--mlp_ratio", default=4, type=int)
        self.parser.add_argument("--save_snapshot", default=200, type=int)
        self.parser.add_argument("--save_weight", default=20, type=int)
        self.parser.add_argument("--random", default=False)  # random select subject
        self.parser.add_argument("--log_path", default='./log/Finetune.log')

    def get_opt(self):
        self.opt = self.parser.parse_args()
        return self.opt
