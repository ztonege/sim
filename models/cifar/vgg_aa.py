# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from antialiased_cnns import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg16_bn_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf2-2c2052a8.pth',
    'vgg16_bn_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf3-1782878a.pth',
    'vgg16_bn_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf4-a1c3559b.pth',
    'vgg16_bn_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf5-c500b52f.pth',
    'vgg16_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf2-60268e0c.pth',
    'vgg16_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf3-e9b0ce42.pth',
    'vgg16_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf4-de9267ac.pth',
    'vgg16_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf5-1391f70c.pth',
    'vgg11_bn_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg11_bn_lpf4_finetune-5d60b5e4.pth',
    'vgg11_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg11_lpf4_finetune-35eab449.pth',
    'vgg13_bn_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg13_bn_lpf4_finetune-45e2a72f.pth',
    'vgg13_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg13_lpf4_finetune-d8ff02c4.pth',
    'vgg16_bn_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf4_finetune-1dd798d1.pth',
    'vgg16_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf4_finetune-79c9dff7.pth',
    'vgg19_bn_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg19_bn_lpf4_finetune-d0114293.pth',
    'vgg19_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg19_lpf4_finetune-7ab2cf45.pth'
}



class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=True, filter_size=1):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers += [nn.MaxPool2d(kernel_size=2, stride=1), BlurPool(in_channels, filt_size=filter_size, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], filter_size=filter_size), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg11_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    print(11111)
    return model


def vgg11_bn(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], filter_size=filter_size, batch_norm=True), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def vgg13(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], filter_size=filter_size), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def vgg13_bn(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], filter_size=filter_size, batch_norm=True), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def vgg16(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], filter_size=filter_size), **kwargs)
    if pretrained:
        if(filter_size==4 and not _force_nonfinetuned):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16_lpf%i'%filter_size], map_location='cpu', check_hash=True)['state_dict'])
    return model


def vgg16_bn(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], filter_size=filter_size, batch_norm=True), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn_lpf%i'%filter_size], map_location='cpu', check_hash=True)['state_dict'])
    return model


def vgg19(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], filter_size=filter_size), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg19_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def vgg19_bn(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], filter_size=filter_size, batch_norm=True), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from antialiased_cnns import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg16_bn_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf2-2c2052a8.pth',
    'vgg16_bn_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf3-1782878a.pth',
    'vgg16_bn_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf4-a1c3559b.pth',
    'vgg16_bn_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf5-c500b52f.pth',
    'vgg16_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf2-60268e0c.pth',
    'vgg16_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf3-e9b0ce42.pth',
    'vgg16_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf4-de9267ac.pth',
    'vgg16_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf5-1391f70c.pth',
    'vgg11_bn_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg11_bn_lpf4_finetune-5d60b5e4.pth',
    'vgg11_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg11_lpf4_finetune-35eab449.pth',
    'vgg13_bn_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg13_bn_lpf4_finetune-45e2a72f.pth',
    'vgg13_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg13_lpf4_finetune-d8ff02c4.pth',
    'vgg16_bn_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_bn_lpf4_finetune-1dd798d1.pth',
    'vgg16_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg16_lpf4_finetune-79c9dff7.pth',
    'vgg19_bn_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg19_bn_lpf4_finetune-d0114293.pth',
    'vgg19_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/vgg19_lpf4_finetune-7ab2cf45.pth'
}



class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, filter_size=1):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers += [nn.MaxPool2d(kernel_size=2, stride=1), BlurPool(in_channels, filt_size=filter_size, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        filter_size (int): [4] Antialiasing filter size
    """

    if pretrained:
        kwargs['init_weights'] = False
    print(cfg['A'])
    model = VGG(make_layers(cfg['A'], filter_size=filter_size), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg11_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def vgg11_bn(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], filter_size=filter_size, batch_norm=True), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def vgg13(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], filter_size=filter_size), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def vgg13_bn(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], filter_size=filter_size, batch_norm=True), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def vgg16(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], filter_size=filter_size), **kwargs)
    if pretrained:
        if(filter_size==4 and not _force_nonfinetuned):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16_lpf%i'%filter_size], map_location='cpu', check_hash=True)['state_dict'])
    return model


def vgg16_bn(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], filter_size=filter_size, batch_norm=True), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn_lpf%i'%filter_size], map_location='cpu', check_hash=True)['state_dict'])
    return model


def vgg19(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], filter_size=filter_size), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg19_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def vgg19_bn(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        filter_size (int): [4] Antialiasing filter size
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], filter_size=filter_size, batch_norm=True), **kwargs)
    if pretrained:
        if(filter_size==4):
            model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn_lpf4_finetune'], map_location='cpu', check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model