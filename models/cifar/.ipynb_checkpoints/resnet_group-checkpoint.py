"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
      x_right_pad = F.pad(x.clone(), p2d_right, "constant", 0)
      x_right_pad = x_right_pad[:, :, :, 1:]
      
      x_right_pad = self.avg_pool_right(x_right_pad)
      x_down_pad = F.pad(x.clone(), p2d_down, "constant", 0)
      x_down_pad = x_down_pad[:, :, 1:, :]
      x_down_pad = self.avg_pool_down(x_down_pad)
      x_right_down_pad = F.pad(x.clone(), p2d_right_down, "constant", 0)
      
      x_right_down_pad = x_right_down_pad[:, :, 1:, 1:]
      x_right_down_pad = self.avg_pool_right_down(x_right_down_pad)
      B, C, H, W= x.shape
      st = torch.stack((x_no_pad, x_right_pad, x_down_pad, x_right_down_pad), dim=2).reshape((B, C*4, H_OUT, W_OUT))
      return st
    

class AA_Group(nn.Module):
    """
    Model definition
    """
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()
        self.test_dense_pool = densePool(2)
        # ceiling
        ceil_channel = (out_channel//in_channel)*in_channel
        if out_channel % in_channel != 0:
          ceil_channel += in_channel 
        self.cnn_group = nn.Conv2d(out_channels=ceil_channel, in_channels=in_channel*4, kernel_size=kernel_size, padding=padding, groups=in_channel)
        self.cnn_aggregate = nn.Conv2d(out_channels=out_channel, in_channels=ceil_channel, kernel_size=1)
    def forward(self, x):
      out = self.test_dense_pool(x)
      out = self.cnn_group(out)
      out = self.cnn_aggregate(out)
      return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                AA_Group(in_channel=in_planes, out_channel=self.expansion * planes, kernel_size=1),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        ####
        self.conv2 = AA_Group(in_channel=planes, out_channel=planes, kernel_size=3, padding=1)
        ####
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # print(out.shape)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18(10)
    y = net(torch.randn(1, 3, 32, 32))


# In[4]:



# In[ ]:





# In[ ]:




