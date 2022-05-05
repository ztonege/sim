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


class densePool(nn.Module):
    """
    Model definition
    """
    def __init__(self, kernel_size, stride_size=2):
        super().__init__()
        self.avg_pool_right = nn.AvgPool2d(kernel_size=kernel_size, stride=stride_size)
        self.avg_pool_down = nn.AvgPool2d(kernel_size=kernel_size, stride=stride_size)
        self.avg_pool_right_down = nn.AvgPool2d(kernel_size=kernel_size, stride=stride_size)
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride_size)

    def forward(self, x):
      N, C, H, W = x.shape
      if W % 2 == 1:
        p2d_right = (0, 1, 0, 0)
        x = F.pad(x, p2d_right, "replicate")
      if H % 2 == 1:
        p2d_down = (0, 0, 0, 1)
        x = F.pad(x, p2d_down, "replicate")
      #no pad
      #pad right
      p2d_right = (0, 1, 0, 0)
      #pad down
      p2d_down = (0, 0, 0, 1)
      #pad right and down
      p2d_right_down = (0, 1, 0, 1)
      x_no_pad = x.clone()
      x_no_pad = self.avg_pool(x_no_pad)
      _, _, H_OUT, W_OUT = x_no_pad.shape
      x_right_pad = F.pad(x.clone(), p2d_right, "replicate")
      x_right_pad = x_right_pad[:, :, :, 1:]
      
      x_right_pad = self.avg_pool_right(x_right_pad)
      x_down_pad = F.pad(x.clone(), p2d_down, "replicate")
      x_down_pad = x_down_pad[:, :, 1:, :]
      x_down_pad = self.avg_pool_down(x_down_pad)
      x_right_down_pad = F.pad(x.clone(), p2d_right_down, "replicate")
      
      x_right_down_pad = x_right_down_pad[:, :, 1:, 1:]
      x_right_down_pad = self.avg_pool_right_down(x_right_down_pad)
      B, C, H, W= x.shape
      st = torch.stack((x_no_pad, x_right_pad, x_down_pad, x_right_down_pad), dim=2).reshape((B, C*4, H_OUT, W_OUT))
      return st
    

class AA_3D(nn.Module):
    """
    Model definition
    """

    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()
        self.test_dense_pool = densePool(2)

        self.conv1 = nn.Conv3d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=(3, kernel_size, kernel_size),
            padding=(0, padding, padding),
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channel * 2, out_channels=out_channel, kernel_size=1
        )

    def forward(self, x):

      B, _, _, _ = x.shape
      out = self.test_dense_pool(x)
      # print("dense shape:", out.shape)
      split_tensor = torch.split(out, 4,dim=1)
      D3_tensor = torch.stack(split_tensor, dim=1)
      conv1_out = self.conv1(D3_tensor)
      conv2_in = conv1_out.view(B, -1, conv1_out.size(3), conv1_out.size(4))
      out = self.conv2(conv2_in)
      return out

class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            AA_3D(in_channel=128, out_channel=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            AA_3D(in_channel=256, out_channel=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            AA_3D(in_channel=512, out_channel=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096),
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
        x = x.reshape(x.size(0), -1)
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

