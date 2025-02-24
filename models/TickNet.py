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
                 stride,
                 use_bottleneck=False):
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
        
        if use_bottleneck:
            # bottleneck_channels = 64  
            self.bottleneck = Bottleneck(in_channels=512, 
                                        bottleneck_channels=256,  
                                        out_channels=128) 
            # self.bottleneck = GroupedBottleneck(in_channels=512, out_channels=128, groups=4)
            # self.bottleneck = MobileNetBottleneck(in_channels=512, out_channels=128)
            
    def forward(self, x):
        residual = x
        x = self.Pw1(x)        
        x = self.Dw(x)        
        x = self.Pw2(x)
        x = self.SE(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:            
            if self.use_bottleneck and self.in_channels > self.out_channels:
                residual = self.bottleneck(residual)
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

class GroupedBottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.groups = groups
        # Convolution 1x1 để giảm số kênh từ 512 -> 64 (nhóm theo groups)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups)
        self.bn1 = torch.nn.BatchNorm2d(out_channels // 2)  # BatchNorm cho conv1
        # Convolution 3x3 để xử lý thông tin (nhóm theo groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn2 = torch.nn.BatchNorm2d(out_channels // 2)  # BatchNorm cho conv2
        # Convolution 1x1 để tăng số kênh từ 64 -> 128 (nhóm theo groups)
        self.conv3 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)  # BatchNorm cho conv3
        self.relu = torch.nn.ReLU(inplace=True)  # Activation

    def forward(self, x):
        residual = x  # Lưu thông tin gốc
        # Giảm số kênh từ 512 -> 64
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Xử lý thông tin với convolution 3x3
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # Tăng số kênh từ 64 -> 128
        x = self.conv3(x)
        x = self.bn3(x)
        # Kết hợp với thông tin gốc (skip connection)
        if residual.shape == x.shape:  # Chỉ cộng nếu kích thước khớp
            x = x + residual
        x = self.relu(x)
        return x

class MobileNetBottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Pointwise Convolution để giảm số kênh từ 512 -> 64
        self.pw1 = torch.nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels // 2)  # BatchNorm cho pw1
        # Depthwise Convolution để xử lý thông tin
        self.dw = torch.nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, groups=out_channels // 2)
        self.bn2 = torch.nn.BatchNorm2d(out_channels // 2)  # BatchNorm cho dw
        # Pointwise Convolution để tăng số kênh từ 64 -> 128
        self.pw2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)  # BatchNorm cho pw2
        self.relu = torch.nn.ReLU(inplace=True)  # Activation

    def forward(self, x):
        residual = x  # Lưu thông tin gốc
        # Giảm số kênh từ 512 -> 64
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Xử lý thông tin với Depthwise Convolution
        x = self.dw(x)
        x = self.bn2(x)
        x = self.relu(x)
        # Tăng số kênh từ 64 -> 128
        x = self.pw2(x)
        x = self.bn3(x)
        # Kết hợp với thông tin gốc (skip connection)
        if residual.shape == x.shape:  # Chỉ cộng nếu kích thước khớp
            x = x + residual
        x = self.relu(x)
        return x
        
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
                use_bottleneck = in_channels > unit_channels * 2
                stage.add_module("unit{}".format(unit_id + 1), FR_PDP_block(in_channels=in_channels, out_channels=unit_channels, stride=stride, use_bottleneck=use_bottleneck))
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
