# 李理的博客
# http://fancyerii.github.io/books/neural-style-transfer/
# 算法：Leon A. Gatys 等人，
# A Neural Algorithm of Artistic Style

# yuchenxi 在源代码基础上作了修改

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import copy

cnn_normalization_mean = torch.Tensor([0.485, 0.456, 0.406]).view((-1, 1, 1))
cnn_normalization_std = torch.Tensor([0.229, 0.224, 0.225]).view((-1, 1, 1))


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # 把mean和std reshape成 [C x 1 x 1]
        # 输入图片是 [B x C x H x W].
        # 因此下面的forward计算可以是用broadcasting
        self.mean = torch.Tensor(mean).view(-1, 1, 1)
        self.std = torch.Tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


imsize = 512
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
saver = transforms.Compose([transforms.ToPILImage()])


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image)
    image = image[0:3].unsqueeze(0)
    return image


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    # a=batch size(=1)
    # b=feature map的数量
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'
                               .format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # 图片的值必须在(0,1)之间，但是梯度下降可能得到这个范围之外的值，所有需要clamp到这个范围里
            input_img.data.clamp_(0, 1)

            # 清空梯度
            optimizer.zero_grad()
            # forward计算，从而得到StyleLoss和ContentLoss
            model(input_img)
            style_score = 0
            content_score = 0

            # 计算Loss
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # Loss乘以weight
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            # 反向求loss对input_img的梯度
            loss.backward()

            run[0] += 1

            if run[0] % 10 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            # 返回loss给LBFGS
            return style_score + content_score

        optimizer.step(closure)
        if run[0] % 10 == 0:
            s = input('continue? [Y/n] : ')
            if s == 'N' or s == 'n':
                break

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('content', help='path of content image')
    parser.add_argument('style', help='path of style image')
    parser.add_argument('output', help='path of output image')
    args = parser.parse_args()
    print('content image: {}'.format(args.content))
    print('style image: {}'.format(args.style))
    print('output image: {}'.format(args.output))

    content_img = image_loader(args.content)
    style_img = image_loader(args.style)
    input_img = content_img.clone()
    model = models.vgg16(pretrained=True).features.eval()
    out_img = run_style_transfer(model, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, style_weight=10000000)
    out_img = saver(out_img[0])
    fp = open(args.output, 'wb')
    out_img.save(fp)
