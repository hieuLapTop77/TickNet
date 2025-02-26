import re
import types

import torch.nn
import torch.nn.init

from .common import conv1x1_block, Classifier,conv3x3_dw_blockAll,conv3x3_block
from .SE_Attention import *
class FR_PDP_block(torch.nn.Module):
    """
    FR_PDP_block for TickNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        self.Pw1 = conv1x1_block(in_channels=in_channels,
                                out_channels=in_channels,                                
                                use_bn=False,
                                activation=None)
        self.Dw = conv3x3_dw_blockAll(channels=in_channels, stride=stride)         
        self.Pw2 = conv1x1_block(in_channels=in_channels,
                                             out_channels=out_channels,                                             
                                             groups=1)
        self.PwR = conv1x1_block(in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SE = SE(out_channels, 16)
    def forward(self, x):
        residual = x
        x = self.Pw1(x)        
        x = self.Dw(x)        
        x = self.Pw2(x)
        x = self.SE(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:                     
            residual = self.PwR(residual)
            x = x + residual
        return x
        
class Bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        return out

class BottleneckSE(nn.Module):
    def __init__(self, in_channels, expansion_factor, out_channels, stride=1, se_ratio=16):
        """
        :param in_channels: Số channel đầu vào (ví dụ 512)
        :param expansion_factor: Hệ số mở rộng (ví dụ 0.5 để chuyển 512->256 nếu cần mở rộng hay giảm)
        :param out_channels: Số channel đầu ra (ví dụ 128)
        :param stride: Stride của depthwise convolution (mặc định 1)
        :param se_ratio: Hệ số giảm của SE block (mặc định 16)
        """
        super(BottleneckSE, self).__init__()
        mid_channels = int(in_channels * expansion_factor)
        
        # Phần mở rộng: Pointwise convolution
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise convolution: xử lý thông tin không gian trên từng channel riêng biệt
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                      padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Giảm số channel: Pointwise convolution để tạo bottleneck
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Thêm SE block sau phần project
        self.se = SE(out_channels, se_ratio)
        
        # Shortcut (nếu điều kiện kích thước thỏa mãn)
        self.use_shortcut = (stride == 1 and in_channels == out_channels)
        if not self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        out = self.se(out)  # Áp dụng SE để tái cân bằng các channel
        if self.use_shortcut:
            out += identity
        else:
            out += self.shortcut(identity)
        return out

        
class TickNet(torch.nn.Module):
    """
    Class for constructing TickNet.    
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1  
                if in_channels == 512 and unit_channels == 128:
                    stage.add_module("unit{}".format(unit_id + 1), BottleneckSE(in_channels=512, expansion_factor=0.5, out_channels=128, stride=stride, se_ratio=16))
                else:
                    stage.add_module("unit{}".format(unit_id + 1), FR_PDP_block(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        self.final_conv_channels = 1024        
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        in_channels = self.final_conv_channels
        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

###
#%% model definitions
###
def build_TickNet(num_classes, typesize='small', cifar=False):
    init_conv_channels = 32
    if typesize=='basic':
        channels = [[128],[64],[128],[256],[512]]
    if typesize=='small':
        channels = [[128],[64,128],[256,512,128],[64,128,256],[512]]
    if typesize=='large':
        channels = [[128],[64,128],[256,512,128,64,128,256],[512,128,64,128,256],[512]]
    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        if typesize=='basic':
            strides = [1, 2, 2, 2, 2]
        else:
            strides = [2, 1, 2, 2, 2]
    return  TickNet(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size)
