import argparse
import os

class Test_Options():
    def __init__(self):
        
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--batch_size", default=10, type=int)
        self.parser.add_argument("--img_size", default=256, type=int)
        self.parser.add_argument("--data_root", default='./data/test/')
        self.parser.add_argument("--slice_num", default=60, type=int)
        
        self.parser.add_argument("--depth", default=12,type=int)
        self.parser.add_argument("--num_workers", default=8, type=int)
        self.parser.add_argument("--data_rate", default=1, type=float)
        self.parser.add_argument("--mae_patch_size", default=8, type=int)
        self.parser.add_argument("--patch_size", default=4, type=int)
        self.parser.add_argument("--encoder_dim", default=128, type=int)
        self.parser.add_argument("--decoder_dim", default=64, type=int)
        self.parser.add_argument("--decoder_depth", default=8, type=int)
        self.parser.add_argument("--decoder_num_heads", default=8, type=int)
        self.parser.add_argument("--vit_dim", default=128, type=int)
        self.parser.add_argument("--window_size", default=8, type=int)
        self.parser.add_argument("--num_heads", default=16, type=int)
        self.parser.add_argument("--mlp_ratio", default=4, type=int)
        self.parser.add_argument("--random", default=False)
        
        self.opt = self.parser.parse_args(args=[])

    def get_opt(self):
        return self.opt
        
