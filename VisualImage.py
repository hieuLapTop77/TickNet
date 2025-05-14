# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:59:56 2023

@author: tuann
"""

import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms, utils

# from models.TickNet import *
# from models.TickNet_only_PDP_for_Visual import *
from models.TickNet import *

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize(mean=0., std=1.)
])
pathout = './imagesample/visual_feature_map'
# pathimage = '../fulldeepwiseMobileNet_CBAMM/imagesample/sample_for_visual_stanford_dogs'
image = Image.open(str('images/n02088364_876.jpg'))

# pathimage = '../fulldeepwiseMobileNet_CBAMM/imagesample/sample_for_visual_ImageNet100'
# image = Image.open(str(pathimage + '/n01558993/ILSVRC2012_val_00041152.JPEG'))
# image = Image.open(str('imagesample/beagle/n02088364_876.jpg'))
# image = Image.open(str('imagesample/bull_mastiff/n02108422_5609.jpg'))

plt.imshow(image)

# model = models.resnet18(pretrained=True)
model = build_TickNet(120, typesize='large_new7', cifar=False)
print(model)

# As you can see above resnet architecture we have a bunch of Conv2d, BatchNorm2d, and ReLU layers. But I want to check only feature maps after Conv2d because this the layer where actual filters were applied. So let’s extract the Conv2d layers and store them into a list also extract corresponding weights and store them in the list as well.

# we will save the conv layer weights in this list
model_weights = []
# we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective wights to the list

for name, layer in model.named_modules():
    print(name, layer)
    if name == 'backbone.init_conv.conv':
        # model_weights.append(layer.weight)
        conv_layers.append(layer)
    if name == 'backbone.stage1':
        # model_weights.append(layer.weight)
        # conv_layers.append(layer)
        conv_layers.append(layer.FR_PDP_block_id1)
        # break;
    if name == 'backbone.stage2':
        # model_weights.append(layer.weight)
        # conv_layers.append(layer)
        conv_layers.append(layer.FR_PDP_block_id2)
        conv_layers.append(layer.FR_PDP_block_id3)
        break
    # if name == 'backbone.stage3':
    #     #model_weights.append(layer.weight)
    #     conv_layers.append(layer)
    #     break;

    # if name == 'backbone.stage4':
    #     #model_weights.append(layer.weight)
    #     conv_layers.append(layer)
    # if name == 'backbone.stage5':
    #     #model_weights.append(layer.weight)
    #     conv_layers.append(layer)
    #     break;
# for i in range(len(model_children)):
#     if type(model_children[i]) == nn.Conv2d:
#         counter+=1
#         model_weights.append(model_children[i].weight)
#         conv_layers.append(model_children[i])
#     elif type(model_children[i]) == nn.Sequential:
#         for j in range(len(model_children[i])):
#             for child in model_children[i][j].children():
#                 if type(child) == nn.Conv2d:
#                     counter+=1
#                     model_weights.append(child.weight)
#                     conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

# Check for GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = model.to(device)
# Apply transformation on the image, Add the batch size, and load on GPU
image = transform(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)

# Process image to every layer and append output and name of the layer to outputs[] and names[] lists
outputs = []
names = []
with torch.no_grad():
    for layer in conv_layers[0:]:
        image = layer(image)
        print('vao day')
        # cv2.imshow('image',np.array(image))
        # cv2.waitKey(0)
        outputs.append(image)
        names.append(str(layer))
print(len(outputs))
# print feature_maps
for feature_map in outputs:
    print(feature_map.shape)

# Now convert 3D tensor to 2D, Sum the same element of every channel
# processed = []
# for feature_map in outputs:
#     feature_map = feature_map.squeeze(0)
#     gray_scale = torch.sum(feature_map,0)
#     gray_scale = gray_scale / feature_map.shape[0]
#     processed.append(gray_scale.data.cpu().numpy())
# Thanh Tuan add start for processed = []
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    # gray_scale = feature_map
    feature_image = gray_scale / feature_map.shape[0]
    feature_image -= feature_image.mean()
    feature_image /= feature_image.std()
    feature_image *= 64
    feature_image += 128
    feature_image = np.clip(feature_image.numpy(), 0, 255).astype('uint8')
    feature_image = torch.tensor(feature_image)
    processed.append(feature_image.data.cpu().numpy())
# Thanh Tuan add end for processed = []
for fm in processed:
    print(fm.shape)

# Plotting feature maps and save
fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str(
    pathout + '/feature_maps_ImageNet100_TickNet_large7.jpg'), bbox_inches='tight')
