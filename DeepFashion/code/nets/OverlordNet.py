import torch 
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import os


model_path = "DeepFashion/VGG16_Global_Cat_Atr.pth"
image_path = "DeepFashion/data/images/img/2-in-1_Space_Dye_Athletic_Tank/img_00000001.jpg"

def get_img_tensor(img_path, use_cuda, get_size=False):
    img = Image.open(img_path)
    original_w, original_h = img.size

    img_size = (224, 224)  # crop image to (224, 224)
    img.thumbnail(img_size, Image.ANTIALIAS)
    img = img.convert('RGB')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    if use_cuda:
        img_tensor = img_tensor.cuda()
    if get_size:
        return img_tensor, original_w, original_w
    else:
        return img_tensor


# model settings
arch = 'vgg'
attribute_num = 26  # num of attributes
category_num = 50  # num of categories
img_size = (224, 224)

model = dict(
    type='GlobalAttrCatePredictor',
    backbone=dict(type='Vgg', layer_setting='vgg16'),
    global_pool=dict(
        type='GlobalPooling',
        inplanes=(7, 7),
        pool_plane=(2, 2),
        inter_channels=[512, 1024],
        outchannels=1024),
    attr_predictor=dict(
        type='AttrPredictor',
        inchannels=1024,
        outchannels=attribute_num,
        loss_attr=dict(
            type='BCEWithLogitsLoss',
            ratio=1,
            weight=None,
            size_average=None,
            reduce=None,
            reduction='mean')),
    cate_predictor=dict(
        type='CatePredictor',
        inchannels=1024,
        outchannels=category_num,
        loss_cate=dict(type='CELoss', ratio=1, weight=None, reduction='mean')),
    pretrained=model_path)


landmark_tensor = torch.zeros(8)

img_tensor = get_img_tensor(image_path,False)
attr_prob, cate_prob = model(
        img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)
