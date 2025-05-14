# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:11:28 2023

@author: tuann
"""
from models.mobilenet_with_FR import *
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet18, resnet50

from models.TickNet import *

# use_cuda = torch.cuda.is_available()
use_cuda = False
if use_cuda:
    print('Using GPU for acceleration')
else:
    print('Using CPU for computation')
    device = torch.device('cpu')

# model = resnet18(pretrained=True)
# target_layers = [model.layer4[-1]]
# model = build_TickNet(120, typesize='small',cifar=False)


imgshow = False
# type_model = 'V2_LiB_Ori'
# #checkpoint = 'V2_CBAM_avg_max_checkpoint_epoch0072_78.31.pth'
# checkpoint = 'V2_CBAM_avg_max_checkpoint_epoch0160_58.09.pth'

# from models.TickNet_only_PDP import *
# type_model = 'TickSmall_PDPOnly'
# checkpoint = 'TickSmall_PDPOnly_checkpoint_epoch0149_57.42.pth'
# model = build_TickNet(120, typesize='large_new',cifar=False)


type_model = 'V2_FRLiB'
checkpoint = 'V2_FRLiB_checkpoint_epoch0178_59.09.pth'
model = build_mobilenet_v2(120, width_multiplier=1.0, cifar=False)

# from models.TickNet_ban_full_reference import *
# type_model = 'TickSmall_FRPDP'
# checkpoint = 'TickSmall_FRPDP_checkpoint_epoch0182_64.54.pth'
# model = build_TickNet(120, typesize='small',cifar=False)

# weights_path = './learnedModels/ImageNet100/' + checkpoint
weights_path = './learnedModels/StanfordDogs/' + checkpoint
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
target_layers = model.backbone.stage5

# arr_foldername = ['n01558993','n02123045','n01855672']
str_path = '../fulldeepwiseMobileNet_CBAMM/imagesample/sample_for_visual_stanford_dogs/'
# arr_foldername = ['African_hunting_dog','Australian_terrier','beagle','bull_mastiff','French_bulldog']
arr_foldername = ['Australian_terrier']
# arr_foldername = ['n02123045']
for foldername in arr_foldername:
    # pathimages = '.\\imagesample\\sample_for_visual_ImageNet100\\' + foldername
    pathtemp = '.\\imagesample\\sample_for_visual_stanford_dogs\\' + foldername
    pathimages = str_path + foldername
    pathout = pathtemp + '_out_' + type_model
    if not os.path.exists(pathout):
        os.makedirs(pathout)
    # list_path = glob.glob(os.path.join(pathimages, "*.JPEG"))
    list_path = glob.glob(os.path.join(pathimages, "*.jpg"))
    for image_path in list_path:
        # image_path = './imagesample/sample_for_visual_ImageNet100/n01855672/n01855672_4197.JPEG'
        filename = Path(image_path).stem + '_' + type_model
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255

        # scale_percent = 220 # percent of original size
        # width = int(rgb_img.shape[1] * scale_percent / 100)
        # height = int(rgb_img.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # dim1 = (1024, 1024)

        # resize image

        # cv2.imshow('image window', rgb_img)
        # # add wait key. window waits until user presses a key
        # cv2.waitKey(0)
        # # and finally destroy/close all open windows
        # cv2.destroyAllWindows()

        # rgb_img = cv2.resize(rgb_img, dim1, interpolation = cv2.INTER_AREA)

        # cv2.imshow('image window', rgb_img)
        # # add wait key. window waits until user presses a key
        # cv2.waitKey(0)
        # # and finally destroy/close all open windows
        # cv2.destroyAllWindows()

        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        # input_tensor = preprocess_image(rgb_img)
        # input_tensor = # Create an input tensor image for your model..
        # Note: input_tensor can be a batch tensor with several images!

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        # cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=True)
        # print ("vao day")
        # You can also use it within a with statement, to make sure it is freed,
        # In case you need to re-create it inside an outer loop:
        # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
        #   ...

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own custom targets
        # That are, for example, combinations of categories, or specific outputs in a non standard model.

        # targets = [ClassifierOutputTarget(281)]
        # print(targets)
        cam.batch_size = 32
        targets = [ClassifierOutputTarget(115)]  # for V1 105 Good TickNet
        # 97 for V2 SRM, MAF
        # targets = [ClassifierOutputTarget(99)]#for V2 CBAM
        # targets = [ClassifierOutputTarget(90)]#for V2 BAM
        # print(targets)
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        print(os.path.join(pathout, filename + '.jpg'))
        cv2.imwrite(os.path.join(pathout, filename + '.jpg'), visualization)

        if imgshow:
            cv2.imshow('image window', visualization)
            # add wait key. window waits until user presses a key
            # cv2.imwrite('./n01855672_10480.jpg', visualization)
            cv2.waitKey(0)
            # and finally destroy/close all open windows
            cv2.destroyAllWindows()
