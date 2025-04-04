import torch.nn as nn
from .common import conv1x1_block, Classifier, conv3x3_dw_blockAll, conv3x3_block
from .SE_Attention import SE

class FR_PDP_block(nn.Module):
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
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)
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
                stage.add_module("unit{}".format(unit_id + 1), FR_PDP_block(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                use_bottleneck = in_channels > unit_channels * 2
                if use_bottleneck:
                    stage.add_module("Bottleneck{}".format(unit_id + 1), Bottleneck(in_channels=512, bottleneck_channels=256, out_channels=128, stride=stride))
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
