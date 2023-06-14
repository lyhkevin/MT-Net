from torch import nn
import torch
from torchsummary import summary
import torch
import torch.nn as nn
import numpy as np
from scipy.signal.windows import gaussian
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F

class Sobel(nn.Module):
    def __init__(self,requires_grad=False):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=requires_grad)
        self.Repad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))

    def forward(self, img):
        x = self.Repad(img)
        x = self.filter(x)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        x[x > 1] = 1
        #x = F.normalize(x,dim=0,p=1)
        return x

class Prewitt(nn.Module):
    def __init__(self,requires_grad=False):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        Gy = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=requires_grad)
        self.Repad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))

    def forward(self, img):
        x = self.Repad(img)
        x = self.filter(x)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        x[x > 1] = 1
        return x

class Canny(nn.Module):
        def __init__(self, threshold=2.0, use_cuda=True):
            # recommend threshold: 2. for image range[0, 1]
            # recommend threshold: 800.0 for image range[0, 255]
            super(Canny, self).__init__()

            self.threshold = threshold
            self.use_cuda = use_cuda

            filter_size = 5
            generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

            self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, filter_size),
                                                        padding=(0, filter_size // 2))
            self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
            self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
            self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size, 1),
                                                      padding=(filter_size // 2, 0))
            self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
            self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

            sobel_filter = np.array([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]])

            self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                                     padding=sobel_filter.shape[0] // 2)
            self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
            self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
            self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                                   padding=sobel_filter.shape[0] // 2)
            self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
            self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

            # filters were flipped manually
            filter_0 = np.array([[0, 0, 0],
                                 [0, 1, -1],
                                 [0, 0, 0]])

            filter_45 = np.array([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, -1]])

            filter_90 = np.array([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, -1, 0]])

            filter_135 = np.array([[0, 0, 0],
                                   [0, 1, 0],
                                   [-1, 0, 0]])

            filter_180 = np.array([[0, 0, 0],
                                   [-1, 1, 0],
                                   [0, 0, 0]])

            filter_225 = np.array([[-1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]])

            filter_270 = np.array([[0, -1, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]])

            filter_315 = np.array([[0, 0, -1],
                                   [0, 1, 0],
                                   [0, 0, 0]])

            all_filters = np.stack(
                [filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

            self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape,
                                                padding=filter_0.shape[-1] // 2)
            self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
            self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

        def forward(self, img):
            if img.shape.__len__() != 4:
                raise ValueError("length of image shape should be 4, that is, image shape should be (N, C, H, W)!")
            if img.shape[1] != 3:
                img = img.repeat(1, 3, 1, 1)
                if img.shape[1] != 3:
                    raise ValueError("Channel of image should be 1 or 3")
            batch_size = img.shape[0]
            img_r = img[:, 0:1]
            img_g = img[:, 1:2]
            img_b = img[:, 2:3]

            blur_horizontal = self.gaussian_filter_horizontal(img_r)
            blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
            blur_horizontal = self.gaussian_filter_horizontal(img_g)
            blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
            blur_horizontal = self.gaussian_filter_horizontal(img_b)
            blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

            blurred_img = torch.stack([blurred_img_r, blurred_img_g, blurred_img_b], dim=1)
            blurred_img = torch.stack([torch.squeeze(blurred_img)])

            grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
            grad_y_r = self.sobel_filter_vertical(blurred_img_r)
            grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
            grad_y_g = self.sobel_filter_vertical(blurred_img_g)
            grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
            grad_y_b = self.sobel_filter_vertical(blurred_img_b)

            # COMPUTE THICK EDGES

            grad_mag = torch.sqrt(grad_x_r ** 2 + grad_y_r ** 2)
            grad_mag += torch.sqrt(grad_x_g ** 2 + grad_y_g ** 2)
            grad_mag += torch.sqrt(grad_x_b ** 2 + grad_y_b ** 2)
            grad_orientation = (
                        torch.atan2(grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b) * (180.0 / 3.14159))
            grad_orientation += 180.0
            grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

            # THIN EDGES (NON-MAX SUPPRESSION)

            all_filtered = self.directional_filter(grad_mag)

            inidices_positive = (grad_orientation / 45) % 8
            inidices_negative = ((grad_orientation / 45) + 4) % 8

            height = inidices_positive.size()[2]
            width = inidices_positive.size()[3]
            pixel_count = height * width
            pixel_range = torch.FloatTensor([range(pixel_count)])
            if self.use_cuda:
                pixel_range = torch.cuda.FloatTensor([range(pixel_count)])
            if batch_size > 1:
                indices = (inidices_positive.view(batch_size, -1).data * pixel_count + pixel_range.repeat(batch_size, 1)).squeeze()

                all_temp = all_filtered.view(batch_size, -1)

                temp = torch.stack((all_temp[0, indices[0].long()], all_temp[1, indices[1].long()]))
                for i in range(2, batch_size):
                    temp = torch.cat((temp, all_temp[i, indices[i].long()].unsqueeze(dim=0)), dim=0)
                channel_select_filtered_positive = temp.view(batch_size, 1, height, width)

                indices = (inidices_negative.view(batch_size, -1).data * pixel_count + pixel_range.repeat(batch_size, 1)).squeeze()

                temp = torch.stack((all_temp[0, indices[0].long()], all_temp[1, indices[1].long()]))
                for i in range(2, batch_size):
                    temp = torch.cat((temp, all_temp[i, indices[i].long()].unsqueeze(dim=0)), dim=0)

                channel_select_filtered_negative = temp.view(batch_size, 1, height, width)
            else:
                indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
                channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1, height, width)

                indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
                channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1, height, width)

            channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

            is_max = channel_select_filtered.min(dim=0)[0] > 0.0
            # is_max = torch.unsqueeze(is_max, dim=0)

            thin_edges = grad_mag.clone()
            if batch_size > 1:
                for i in range(batch_size):
                    thin_edges[i, is_max[i] == 0] = 0.0
            else:
                is_max = torch.unsqueeze(is_max, dim=0)
                thin_edges[is_max == 0] = 0.0
            # THRESHOLD
            thresholded = thin_edges.clone()
            thresholded[thin_edges < self.threshold] = 0.0

            early_threshold = grad_mag.clone()
            early_threshold[grad_mag<self.threshold] = 0.0

            assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()
            thresholded[thresholded >= 1] = 1

            return thresholded


if __name__ == '__main__':
    canny = Canny()
    #img = Image.open('./lena_gray.png').convert('RGB')
    #img_size = (224,224)
    #img_transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    #img = img_transform(img).unsqueeze(dim=0)
    img = torch.zeros((10,1,224,224))
    output = canny(img)
    print(output.size())
    
