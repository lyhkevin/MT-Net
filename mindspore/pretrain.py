from utils.pretrain_dataloader import *
from model.EdgeMAE import *
import argparse
from utils.pretrain_dataloader import *
from mindspore import dtype as mstype
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, nn, ops

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument("--batch_size", default=5, type=int)
        self.parser.add_argument("--epoch", default=100, type=int)
        self.parser.add_argument("--patch_size", default=8, type=int)
        self.parser.add_argument("--img_size", default=(256, 256), type=int)
        self.parser.add_argument("--modality", default='all')  # using all modalities for pre-training (t1, t2, t1c, flair)
        self.parser.add_argument("--masking_ratio", default=0.7, type=float)
        self.parser.add_argument("--num_workers", default=8, type=int)
        self.parser.add_argument('--img_save_path', type=str, default='./snapshot/EdgeMAE/')
        self.parser.add_argument('--weight_save_path', type=str, default='./weight/EdgeMAE/')
        self.parser.add_argument("--data_root", default='./data/train/')

        self.parser.add_argument("--depth", default=12, type=int)
        self.parser.add_argument("--decoder_depth", default=8, type=int)
        self.parser.add_argument("--save_output", default=500, type=int)
        self.parser.add_argument("--save_weight", default=5, type=int)
        self.parser.add_argument("--num_heads", default=16, type=int)
        self.parser.add_argument("--decoder_num_heads", default=8, type=int)
        self.parser.add_argument("--mlp_ratio", default=4, type=int)
        self.parser.add_argument("--dim_encoder", default=128, type=int)
        self.parser.add_argument("--dim_decoder", default=64, type=int)
        self.parser.add_argument("--log_path", default='./log/single_gpu.log')

    def get_opt(self):
        self.opt = self.parser.parse_args(args=[])
        return self.opt

def forward_fn(img, edge_img, mask):
    loss_img, loss_edge, pred_img, pred_edge = model(img, edge_img, mask)
    loss = loss_img + 0.1 * loss_edge
    return loss, pred_img, pred_edge

if __name__ == '__main__':
    opt = Options().get_opt()
    num_patches = (opt.img_size[0] // opt.patch_size) ** 2
    loader = Pretrain_Dataset(img_size=opt.img_size, image_root=opt.data_root, modality='all')
    dataset = GeneratorDataset(source=loader, column_names=['img', 'edge_img'])
    batch_size = opt.batch_size
    dataset = dataset.batch(batch_size)
    os.makedirs(opt.img_save_path, exist_ok=True)
    os.makedirs(opt.weight_save_path, exist_ok=True)

    model = EdgeMAE(image_size=opt.img_size[0], patch_size=opt.patch_size, in_channels=1, embed_dim=768, depth=12, num_heads=12,
                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,act_layer=partial(nn.GELU, approximate=False),
                    norm_layer=partial(nn.LayerNorm, epsilon=1e-6))
    optimizer = nn.Adam(model.trainable_params(), learning_rate=opt.lr)
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    for epoch in range(0,opt.epoch):
        iteration = 0
        for batch in dataset.create_dict_iterator():

            iteration = iteration + 1
            img, edge_img = batch['img'], batch['edge_img']
            B, C, H, W = img.shape

            mask = get_mask(size=num_patches, mask_rate=opt.masking_ratio, batch_size=opt.batch_size)
            (loss, pred_img, pred_edge), grads = grad_fn(img, edge_img, mask)
            optimizer(grads)

            loss = loss.asnumpy()
            print("[Epoch %d/%d] [Batch %d] [loss: %f]" % (epoch, opt.epoch, iteration, loss))

            # if iteration % 100 == 0:
            #     img,edge_img = img[0][0].asnumpy() * 255,edge_img[0][0].asnumpy() * 255
            #     img,edge_img = Image.fromarray(img).convert("L"),Image.fromarray(edge_img).convert("L")
            #     img.save(os.path.join(opt.img_save_path, '{}_{}_img.png'.format(epoch, iteration)))
            #     edge_img.save(os.path.join(opt.img_save_path, '{}_{}_edge_img.png'.format(epoch, iteration)))

        if epoch % 10 == 0:
            mindspore.save_checkpoint(model, 'model' + str(epoch) + '.ckpt')
    mindspore.save_checkpoint(model, "model.ckpt")









