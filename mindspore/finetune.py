from model.EdgeMAE_finetune import *
from model.TransUNet import *
import argparse
from utils.finetune_dataloader import *
from mindspore import dtype as mstype
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, nn, ops
import logging

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument("--batch_size", default=4, type=int)
        self.parser.add_argument("--epoch", default=100, type=int)
        self.parser.add_argument("--patch_size", default=8, type=int)
        self.parser.add_argument("--img_size", default=(256, 256), type=int)
        self.parser.add_argument("--consistency_loss", default=1000)
        self.parser.add_argument("--source_modality", default='t1')
        self.parser.add_argument("--target_modality", default='t2')
        self.parser.add_argument("--augment", default=True)
        self.parser.add_argument("--decay_epoch", default=50, type=int)
        self.parser.add_argument("--decay_rate", default=0.1, type=int)
        self.parser.add_argument('--img_save_path', type=str, default='../snapshot/finetune/')
        self.parser.add_argument('--weight_save_path', type=str, default='../weight/finetune/')
        self.parser.add_argument('--pretrain_path', type=str, default='../weight/pretrain/EdgeMAE.ckpt')
        self.parser.add_argument("--data_root", default='../data/train/')
        self.parser.add_argument("--save_output", default=500, type=int)
        self.parser.add_argument("--save_weight", default=5, type=int)
        self.parser.add_argument("--log_path", default='./log/finetune.log')

    def get_opt(self):
        self.opt = self.parser.parse_args(args=[])
        return self.opt

def forward_fn(source_img, target_img):
    loss = nn.L1Loss(reduction='mean')
    pred = G(source_img)
    features_pred = FC_module(pred)
    features_gt = FC_module(target_img)
    fc_loss = 0
    for j in range(len(features_pred)):
        fc_loss = fc_loss + loss(features_pred[j], features_gt[j])
    synthesis_loss = loss(pred, target_img)
    loss = synthesis_loss + fc_loss
    return loss, synthesis_loss, fc_loss, pred

if __name__ == '__main__':

    opt = Options().get_opt()
    dataset = Finetune_Dataset(img_size=opt.img_size, image_root=opt.data_root, source_modality='t1', target_modality='t2', augment=opt.augment)
    dataloader = GeneratorDataset(source=dataset, column_names=['source', 'target'])
    batch_size = opt.batch_size
    dataloader = dataloader.batch(batch_size)
    os.makedirs(opt.img_save_path, exist_ok=True)
    os.makedirs(opt.weight_save_path, exist_ok=True)

    logging.basicConfig(filename=opt.weight_save_path + 'finetune.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    EdgeMAE = EdgeMAE_finetune(image_size=opt.img_size[0], patch_size=opt.patch_size, in_channels=1, embed_dim=384, depth=12, num_heads=12,
                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,act_layer=partial(nn.GELU, approximate=False),
                    norm_layer=partial(nn.LayerNorm, epsilon=1e-6))
    TransUNet = TransUNet(patch_size=8, in_channels=1, embed_dim=384, num_heads=16, num_classes=1)
    param_dict = load_checkpoint(opt.pretrain_path)
    load_param_into_net(EdgeMAE, param_dict)
    print('successfully load params')
    G = Generator(EdgeMAE, TransUNet)
    FC_module = EdgeMAE_finetune(image_size=opt.img_size[0], patch_size=opt.patch_size, in_channels=1, embed_dim=384, depth=12, num_heads=12,
                    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,act_layer=partial(nn.GELU, approximate=False),
                    norm_layer=partial(nn.LayerNorm, epsilon=1e-6))
    FC_model_parameters = FC_module.get_parameters()
    for parameter in FC_model_parameters:
        parameter.requires_grad = False
    optimizer = nn.Adam(G.trainable_params(), learning_rate=opt.lr)
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    for epoch in range(0, opt.epoch):

        iteration = 0
        decay = opt.decay_rate ** (epoch // opt.decay_epoch)
        optimizer.get_lr().set_data(opt.lr * decay)

        for batch in dataloader.create_dict_iterator():

            iteration = iteration + 1
            source_img, target_img = batch['source'], batch['target']
            (loss, synthesis_loss, fc_loss, pred), grads = grad_fn(source_img, target_img)
            optimizer(grads)

            loss = loss.asnumpy()
            print("[Epoch %d/%d] [Batch %d] [loss: %f] [synthesis_loss: %f] [fc_loss: %f]" % (epoch, opt.epoch, iteration, loss, synthesis_loss, fc_loss))

            #if iteration % 100 == 0:
            pred, target_img = pred[0][0].asnumpy() * 255, target_img[0][0].asnumpy() * 255
            pred, target_img = pred.astype(np.uint8), target_img.astype(np.uint8)
            pred, target_img = Image.fromarray(pred).convert("L"),Image.fromarray(target_img).convert("L")
            pred.save(os.path.join(opt.img_save_path, '{}_{}_img.png'.format(epoch, iteration)))
            target_img.save(os.path.join(opt.img_save_path, '{}_{}_gt.png'.format(epoch, iteration)))
            logging.info("[Epoch %d/%d] [Batch %d] [loss: %f] [synthesis_loss: %f] [fc_loss: %f]" % (epoch, opt.epoch, iteration, loss, synthesis_loss, fc_loss))

        if epoch % 10 == 0:
            mindspore.save_checkpoint(G, opt.weight_save_path + 'model' + str(epoch) + '.ckpt')

    mindspore.save_checkpoint(G, opt.weight_save_path + "model.ckpt")