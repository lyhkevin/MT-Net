import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal

class ConvTranspose2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, pad_mode):
        super(ConvTranspose2d, self).__init__()
        self.conv2d_transpose = P.Conv2DTranspose(out_channel=out_channels, kernel_size=kernel_size,
                                                  stride=stride, pad_mode=pad_mode, pad=padding,
                                                  dilation=1, group=groups, output_padding=output_padding)

    def construct(self, x):
        return self.conv2d_transpose(x)


class DSF(nn.Cell):
    def __init__(self, dim, input_resolution):
        super(DSF, self).__init__()
        self.r = input_resolution
        self.upconv = ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2,
                                      padding=0, output_padding=0, groups=dim, pad_mode='pad')
        self.conv1 = nn.Conv2d(2 * dim, out_channels=1, kernel_size=1, has_bias=True)
        self.conv2 = nn.Conv2d(2 * dim, out_channels=1, kernel_size=1, has_bias=True)
        self.pooling = nn.AvgPool2d(kernel_size=self.r)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(axis=4)

        self.reshape = P.Reshape()
        self.cat = P.Concat(axis=1)
        self.mul = P.Mul()

    def construct(self, x, x1, x2):
        x1 = self.reshape(x1, (-1, self.dim, int(0.5 * self.r), int(0.5 * self.r)))
        x1 = self.upconv(x1)
        x2 = self.reshape(x2, (-1, self.dim, self.r, self.r))
        x = self.reshape(x, (-1, self.dim, self.r, self.r))

        s1 = self.cat((x1, x))
        s1 = self.conv1(s1)
        s1 = self.sigmoid(s1)
        x1 = self.mul(x1, s1)

        s2 = self.cat((x2, x))
        s2 = self.conv2(s2)
        s2 = self.sigmoid(s2)
        x2 = self.mul(x2, s2)

        c = self.cat((x1, x2))
        c = self.pooling(c)
        c = self.reshape(c, (-1, 2, self.dim, 1, 1))
        c = self.softmax(c)
        c1, c2 = c[..., 0], c[..., 1]

        x = self.mul(c1, x1) + self.mul(c2, x2)
        x = self.reshape(x, (-1, self.r * self.r, self.dim))
        return x
