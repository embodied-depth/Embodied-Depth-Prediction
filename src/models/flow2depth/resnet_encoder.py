# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

def depth_to_disp(_depth, min_depth, max_depth):
    depth = torch.clamp(_depth, min_depth, max_depth)
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp_ = (scaled_disp - min_disp) / (max_disp - min_disp)
    return disp_

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1, input_depth=False):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.input_depth = input_depth
        if input_depth:
            self.conv1 = nn.Conv2d(
                num_input_images * 4, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                num_input_images * 3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, input_depth=False):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, input_depth=input_depth)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        if input_depth:
            loaded['conv1.weight'] = torch.cat([loaded['conv1.weight'], torch.randn((64, 1, 7, 7 ))], dim=1)
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, input_depth=False):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.input_depth = input_depth

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, input_depth=input_depth)
        else:
            self.encoder = resnets[num_layers](pretrained)
            if self.input_depth:
                self.encoder.conv1 =  nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if self.input_depth:
            bias = torch.tensor([0.485, 0.456, 0.406, 1.565]).reshape(1,4, 1,1).repeat(1, num_input_images, 1, 1)
            sigma = torch.tensor([0.229, 0.224, 0.225, 1.761]).reshape(1,4, 1,1).repeat(1,num_input_images , 1, 1)

        else:
            bias = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3, 1,1).repeat(1, num_input_images, 1, 1)
            sigma = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3, 1,1).repeat(1,num_input_images , 1, 1)

        #self.register_buffer('bias', bias)
        #self.register_buffer('sigma', sigma)

        self.bias = bias
        self.sigma = sigma

    def forward(self, input_image):
        self.features = []
        #if self.bias == None:
        #    if self.input_depth:
        #        self.bias = torch.tensor([0.485, 0.456, 0.406, 1.565]).reshape(1,4, 1,1).to(input_image.device)
        #        self.sigma = torch.tensor([0.229, 0.224, 0.225, 1.761]).reshape(1,4, 1,1).to(input_image.device)
        #    else:
        #        self.bias = torch.tensor([0.485, 0.456, 0.406 ]).reshape(1,3, 1,1).to(input_image.device)
        #        self.sigma = torch.tensor([0.229, 0.224, 0.225 ]).reshape(1,3, 1,1).to(input_image.device)

        x = (input_image - self.bias.to(input_image.device)) / self.sigma.to(input_image.device)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class MultiInputResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(MultiInputResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512]) * num_input_images

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))


        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        bias = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3, 1,1)
        sigma = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3, 1,1)

        self.bias = bias
        self.sigma = sigma

    def forward(self, input_image):
        '''
        Input:
            input_image [B, T, C, H, W] 
        Output:
            features [B, T, c, h, w]
        '''
        assert input_image.dim() == 5, 'Wrong input dims should be 5, but received {}'.format(input_image.dim())
        self.features = []

        B, T, _, _, _ = input_image.shape
        input_image = input_image.flatten(0,1)
        x = (input_image - self.bias.to(input_image.device)) / self.sigma.to(input_image.device)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        for i in range(len(self.features)):
            _, c, h, w = self.features[i].shape
            self.features[i] = self.features[i].reshape(B, T, c, h, w)

        return self.features

class MultiInputDepthEncoder(nn.Module):
    def __init__(self, num_layers, num_input_images=1):
        super(MultiInputDepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512]) * num_input_images

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        pretrained = False

        self.encoder = resnets[num_layers](pretrained)
        self.encoder.conv1 =  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        bias  = torch.tensor([ 0.]).reshape(1,1, 1,1)
        sigma = torch.tensor([ 1.]).reshape(1,1, 1,1)
        self.bias = bias
        self.sigma = sigma

    def forward(self, input_image):
        '''
        Input:
            input_image [B, T, C, H, W] 
        Output:
            features [B, T, c, h, w]
        '''
        assert input_image.dim() == 5, 'Wrong input dims should be 5, but received {}'.format(input_image.dim())
        self.features = []

        B, T, _, _, _ = input_image.shape
        input_image = input_image.flatten(0,1)
        x = (input_image - self.bias.to(input_image.device)) / self.sigma.to(input_image.device)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        for i in range(len(self.features)):
            _, c, h, w = self.features[i].shape
            self.features[i] = self.features[i].reshape(B, T, c, h, w)

        return self.features

# Cat RGB embedding and Depth embedding
class RGBDEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1, input_depth=False):
        super(RGBDEncoder, self).__init__()
        self.input_depth = input_depth
        self.rgb_encoder = MultiInputResnetEncoder(num_layers, pretrained, num_input_images)
        if self.input_depth:
            self.d_encoder  = MultiInputDepthEncoder(num_layers, num_input_images)
            self.num_ch_enc = [ i+j for i, j in zip(self.rgb_encoder.num_ch_enc, self.d_encoder.num_ch_enc)]
        else:
            self.num_ch_enc = self.rgb_encoder.num_ch_enc

    def forward(self, rgb, d=None):
        rgb_f = self.rgb_encoder(rgb)
        if self.input_depth:
            d_f   = self.d_encoder(d)
            out_ = []
            for f1, f2 in zip(rgb_f, d_f):
                out_.append(torch.cat([f1, f2], dim=2))
        else:
            out_ = rgb_f

        return out_

# Add RGB embedding and Depth embedding
class RGBDEncoderv2(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1, input_depth=False):
        super(RGBDEncoderv2, self).__init__()
        self.input_depth = input_depth
        self.rgb_encoder = MultiInputResnetEncoder(num_layers, pretrained, num_input_images)
        if self.input_depth:
            self.d_encoder  = MultiInputDepthEncoder(num_layers, num_input_images)
        self.num_ch_enc = self.rgb_encoder.num_ch_enc

    def forward(self, rgb, d=None):
        rgb_f = self.rgb_encoder(rgb)
        if self.input_depth:
            d_f   = self.d_encoder(d)
            out_ = []
            for f1, f2 in zip(rgb_f, d_f):
                out_.append(f1 + f2)
        else:
            out_ = rgb_f

        return out_

if __name__ == '__main__':
    net = ResnetEncoder(18, True, 5, False)