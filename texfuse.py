# ==================================================================
# TexFuse 
#
# Copyright (C) 2021 argenycw <argencwyan@gmail.com>
# under WTFPL v2 Public License
# ==================================================================

import os
from os.path import isfile, join

import argparse
import numpy as np

from PIL import Image
import torch
import torch.nn.functional as F
#import torchvision.models as models
from model import *
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd.gradcheck import zero_gradients

import warnings
warnings.filterwarnings("ignore") # suppress UserWarnings

def capture_feature(module, input, output):
    global intermediate_features
    intermediate_features.append(output)

# return a model with reduction in modules/layers
# so far hard-coded for VGG16
def vgg16_manipulate(vgg16):
    vgg16.features = vgg16.features[:16]
    vgg16.avgpool = torch.nn.Identity()
    vgg16.classifier = torch.nn.Identity()
    return vgg16

# [unknown, unknown] => [0, 1]
def unnormalize(img, mean, std):
    img[0] = img[0] * std[0] + mean[0]
    img[1] = img[1] * std[1] + mean[1]
    img[2] = img[2] * std[2] + mean[2]
    if std == [1, 1, 1]:
        img /= 255.0
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceFuse by AG')
    parser.add_argument('--model', default='vgg', type=str, help='The pretrained model to pick from (vgg/resnet)') 
    parser.add_argument('--source', default='', type=str, help='the path of the source image')
    parser.add_argument('--target', default='', type=str, help='the path of the target image') 
    parser.add_argument('--eps', default=10, type=int, help='Maximum perturbation strength (over 255)') 
    parser.add_argument('--iter', default=50, type=int, help='Number of iterations for projected gradient computation') 
    parser.add_argument('--step_size', default=2, type=int, help='Please just keep it as default, thanks.') 
    parser.add_argument('-o', '--out', default='attack.png', type=str, help='The output filename for the attacked image.') 
    parser.add_argument('-u', '--upsample', action='store_true', help='Upsample the perturbation rather than doublesample the output.')
    args = parser.parse_args()

    # load a pretrained models from VGGFace or VGGFace2
    try:
        if args.model == 'vgg':
            net = vgg_face_dag(weights_path="pretrained/vgg_face_dag.pth")
            meta = net.meta
        elif args.model == 'resnet':
            net = resnet50_scratch_dag(weights_path="pretrained/resnet50_scratch_dag.pth")
            meta = net.meta
        else:
            print('Undefined model type: %s' % args.model)
            exit(0)
    except FileNotFoundError as err:
        print("File not found: ", err)
        print("Please make sure you have downloaded the pretrained models specified in readme.md")
        exit(0)

    # input validation
    if args.source == '' or args.target == '':
        print('You must specify the source and target image via --source and --target.')
        exit(0)
    if args.eps < 1 or args.eps > 255:
        print('Invalid value of eps: %d (acceptable range: 1 - 255)' % args.eps)
        exit(0)
    
    # load sample images
    source_img_path = args.source
    target_img_path = args.target    

    source_raw = Image.open(source_img_path).convert("RGB")
    target_raw = Image.open(target_img_path).convert("RGB")
    source_w, source_h = source_raw.size
    target_w, target_h = target_raw.size

    img_preprocess = compose_transforms(meta)

    if abs(1 - ((source_w / target_w) / (source_h / target_h))) > 0.1:
        print('Warning: A huge difference between dimensions in source and target is detected. Note the image may be stretched to a certain degree.')
    if (source_w != source_h):
        print('Warning: Non-square source image input. The final output may possibly be stretched.')

    source_img = img_preprocess(source_raw).unsqueeze(0)
    target_img = img_preprocess(target_raw).unsqueeze(0)

    # to capture the intial features
    source_feature = net(source_img)
    target_feature = net(target_img)

    num_iter = args.iter
    scale = source_img.max() - source_img.min()
    eps = args.step_size / 255.0 * scale
    max_eps = args.eps / 255.0 * scale

    adv_img = source_img.clone() + torch.zeros_like(source_img).uniform_(-eps, eps)
    for i in range(num_iter):
        adv_img.requires_grad_()
        zero_gradients(adv_img)

        feature = net(adv_img)

        loss = F.l1_loss(feature, target_feature)
        loss.backward(retain_graph=True)

        adv_img = adv_img.data - eps * torch.sign(adv_img.grad.data)

        adv_img = torch.min(torch.max(adv_img, source_img - max_eps), source_img + max_eps)
        # log progress
        print('Iter: %d / %d. Feature Similarity: %f' % (i+1, num_iter, loss.item()), end='\r')        
    print()
    
    # upsample the perturbation when set
    if args.upsample:
        '''
        delta = adv_img - source_img
        delta_large = F.interpolate(delta, size=(source_h, source_w), mode='bicubic', align_corners=False)
        uptemptransform = compose_transforms(meta, to_resize=False)
        adv_img = uptemptransform(source_raw) + delta_large
        '''
        print('Upsample: (%d, %d) => (%d, %d)' % (adv_img.shape[-1], adv_img.shape[-2], source_w, source_h))
        adv_img = F.interpolate(adv_img, size=(source_h, source_w), mode='bicubic', align_corners=False)

    outfile = join('output', args.out)
    save_image(unnormalize(adv_img.squeeze(0), meta['mean'], meta['std']), outfile)
    print("Attack Completed. Example is saved as %s" % outfile)
