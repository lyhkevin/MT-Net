from model.EdgeMAE_finetune import *
from model.TransUNet import *
import argparse
from utils.finetune_dataloader import *
from mindspore import dtype as mstype
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, nn, ops
import logging
from math import log10, sqrt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def psnr(res,gt):
    mse = np.mean((res - gt) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def mse(res,gt):
    mae = np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
    return mae

def nmse(res,gt):
    Norm = np.linalg.norm((gt * gt),ord=2)
    if np.all(Norm == 0):
        return 0
    else:
        nmse = np.linalg.norm(((res - gt) * (res - gt)),ord=2) / Norm
    return nmse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--batch_size", default=5, type=int)
        self.parser.add_argument("--patch_size", default=8, type=int)
        self.parser.add_argument("--img_size", default=(256, 256), type=int)
        self.parser.add_argument("--augment", default=False)
        self.parser.add_argument("--slice_num", default=60)
        self.parser.add_argument("--source_modality", default='t1')
        self.parser.add_argument("--target_modality", default='t2')
        self.parser.add_argument('--img_save_path', type=str, default='../snapshot/test/')
        self.parser.add_argument('--weight_path', type=str, default='../weight/finetune/G.ckpt')
        self.parser.add_argument("--data_root", default='../data/train/')

    def get_opt(self):
        self.opt = self.parser.parse_args(args=[])
        return self.opt

if __name__ == '__main__':
    opt = Options().get_opt()
    dataset = Finetune_Dataset(img_size=opt.img_size, image_root=opt.data_root, source_modality='t1', target_modality='t2', augment=opt.augment)
    dataloader = GeneratorDataset(source=dataset, column_names=['source', 'target'])
    batch_size = opt.batch_size
    dataloader = dataloader.batch(batch_size)
    os.makedirs(opt.img_save_path, exist_ok=True)

    EdgeMAE = EdgeMAE_finetune(image_size=opt.img_size[0], patch_size=opt.patch_size, in_channels=1, embed_dim=384, depth=12, num_heads=12,
                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,act_layer=partial(nn.GELU, approximate=False),
                    norm_layer=partial(nn.LayerNorm, epsilon=1e-6))
    TransUNet = TransUNet(patch_size=8, in_channels=1, embed_dim=384, num_heads=16, num_classes=1)
    print('successfully load params')
    G = Generator(EdgeMAE, TransUNet)
    param_dict = load_checkpoint(opt.weight_path)
    load_param_into_net(G, param_dict)
    G_parameters = G.get_parameters()

    PSNR = []
    NMSE = []
    SSIM = []

    for parameter in G_parameters:
        parameter.requires_grad = False

    iteration = 0
    for batch in tqdm(dataloader.create_dict_iterator()):
        iteration = iteration + 1
        source_img, target_img = batch['source'], batch['target']
        batch_size = source_img.shape[0]
        pred = G(source_img)

        for j in range(batch_size):
            img_pred, img_target = pred[j][0].asnumpy() * 255, target_img[j][0].asnumpy() * 255
            img_pred, img_target = img_pred.astype(np.uint8), img_target.astype(np.uint8)

            PSNR.append(psnr(img_pred, img_target))
            NMSE.append(nmse(img_pred, img_target))
            SSIM.append(ssim(img_pred, img_target))

            img_pred, img_target = Image.fromarray(img_pred).convert("L"), Image.fromarray(img_target).convert("L")
            img_pred.save(os.path.join(opt.img_save_path, '{}_img.png'.format(iteration)))
            img_target.save(os.path.join(opt.img_save_path, '{}_gt.png'.format(iteration)))

    PSNR = np.asarray(PSNR)
    NMSE = np.asarray(NMSE)
    SSIM = np.asarray(SSIM)

    PSNR = PSNR.reshape(-1, opt.slice_num)
    NMSE = NMSE.reshape(-1, opt.slice_num)
    SSIM = SSIM.reshape(-1, opt.slice_num)

    PSNR = np.mean(PSNR, axis=1)
    NMSE = np.mean(NMSE, axis=1)
    SSIM = np.mean(SSIM, axis=1)

    print("PSNR mean:", np.mean(PSNR), "PSNR std:", np.std(PSNR))
    print("NMSE mean:", np.mean(NMSE), "NMSE std:", np.std(NMSE))
    print("SSIM mean:", np.mean(SSIM), "SSIM std:", np.std(SSIM))

