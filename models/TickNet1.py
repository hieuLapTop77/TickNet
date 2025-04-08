import torch.nn as nn
from .common import conv1x1_block, Classifier, conv3x3_dw_blockAll, conv3x3_block
from .SE_Attention import SE
import re
import types
from collections import OrderedDict

import torch.nn
import torch.nn.init
class FR_PDP_block(torch.nn.Module):
    """
    FR_PDP_block for TickNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()
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
    """
    Bottleneck Block với Depthwise Separable Convolution thay cho Conv 3x3 chuẩn.
    Cấu trúc:
    1. Conv 1x1 (giảm kênh) + BN + ReLU
    2. Depthwise Conv 3x3 + BN + ReLU
    3. Pointwise Conv 1x1 + BN
    4. Cộng với kết nối tắt (residual/shortcut)
    5. ReLU cuối cùng
    """
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels # Lưu lại để dùng trong downsample

        # --- Nhánh chính ---
        # 1. Lớp Conv 1x1 đầu tiên: Giảm số kênh, áp dụng stride nếu có
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu1 = nn.ReLU(inplace=True) # Đặt tên riêng cho ReLU

        # 2. Depthwise Separable Convolution (thay thế cho Conv 3x3 chuẩn)
        # 2a. Depthwise Conv 3x3
        self.dw_conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels, # out_channels = in_channels
            kernel_size=3,
            stride=1, # Stride thường áp dụng ở conv1 hoặc downsample
            padding=1,
            groups=bottleneck_channels, # Điểm mấu chốt của depthwise conv
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=True) # Đặt tên riêng

        # 2b. Pointwise Conv 1x1 (sau depthwise)
        self.pw_conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels, # Output channels của bước này vẫn là bottleneck_channels
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(bottleneck_channels) # BN sau pointwise conv
        self.relu3 = nn.ReLU(inplace=True) # ReLU sau pointwise conv + BN

        # 3. Lớp Conv 1x1 cuối cùng: Khôi phục/tăng số kênh lên out_channels
        #    Đầu vào là bottleneck_channels (từ pointwise conv)
        self.conv4 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels) # BN cuối cùng của nhánh chính

        # --- Kết nối tắt (Shortcut/Residual Connection) ---
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels))
            ]))

        # ReLU cuối cùng (sau khi cộng residual)
        self.relu_final = nn.ReLU(inplace=True)

        # In thông tin tham số (chỉ để debug, có thể xóa)
        # params_dw = sum(p.numel() for p in self.dw_conv2.parameters())
        # params_pw = sum(p.numel() for p in self.pw_conv2.parameters())
        # print(f"BottleneckDW: DWConv params={params_dw}, PWConv params={params_pw}, Total DW+PW={params_dw+params_pw}")


    def forward(self, x):
        residual = x

        # Nhánh chính
        # 1. Conv 1x1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # 2. Depthwise Separable Conv
        # 2a. Depthwise 3x3
        out = self.dw_conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # 2b. Pointwise 1x1
        out = self.pw_conv2(out)
        out = self.bn3(out)
        out = self.relu3(out) # Có thể bỏ ReLU ở đây nếu muốn giống một số cấu trúc khác

        # 3. Conv 1x1 cuối
        out = self.conv4(out)
        out = self.bn4(out) # BN cuối cùng trước khi cộng

        # Xử lý kết nối tắt
        if self.downsample is not None:
            residual = self.downsample(x)

        # Cộng kết nối tắt
        out += residual

        # ReLU cuối cùng
        out = self.relu_final(out)

        return out
        
class TickNet(nn.Module):
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_strides,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = nn.Sequential()
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", nn.BatchNorm2d(num_features=in_channels))
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_strides))

        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                if in_channels == 512 and unit_channels == 128:
                    stage.add_module("Bottleneck{}".format(unit_id + 1), Bottleneck(in_channels=512, bottleneck_channels=256, out_channels=128, stride=stride))
                else:
                    stage.add_module("unit{}".format(unit_id + 1), FR_PDP_block(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)

        self.final_conv_channels = 1024
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        self.backbone.add_module("global_pool", nn.AdaptiveAvgPool2d(output_size=1))
        in_channels = self.final_conv_channels

        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)
        self.init_params()

    def init_params(self):
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def build_TickNet(num_classes, typesize='small', cifar=False):
    init_conv_channels = 32
    if typesize == 'basic':
        channels = [[128], [64], [128], [256], [512]]
    elif typesize == 'small':
        channels = [[128], [64, 128], [256, 512, 128], [64, 128, 256], [512]]
    elif typesize == 'large':
        channels = [[128], [64, 128], [256, 512, 128, 64, 128, 256], [512, 128, 64, 128, 256], [512]]

    if cifar:
        in_size = (32, 32)
        init_conv_strides = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_strides = 2
        if typesize == 'basic':
            strides = [1, 2, 2, 2, 2]
        else:
            strides = [2, 1, 2, 2, 2]

    return TickNet(num_classes=num_classes,
                   init_conv_channels=init_conv_channels,
                   init_conv_strides=init_conv_strides,
                   channels=channels,
                   strides=strides,
                   in_size=in_size)
