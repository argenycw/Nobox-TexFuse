# ==================================================================
# FaceFuse 
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
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd.gradcheck import zero_gradients

intermediate_features = []
def capture_feature(module, input, output):
    global intermediate_features
    intermediate_features.append(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceFuse by AG')
    parser.add_argument('--source', default='', type=str, help='the path of the source image')
    parser.add_argument('--target', default='', type=str, help='the path of the target image') 
    parser.add_argument('--eps', default=50, type=int, help='Maximum perturbation strength (over 255)') 
    parser.add_argument('--iter', default=50, type=int, help='Number of iterations for projected gradient computation') 
    parser.add_argument('--step_size', default=5, type=int, help='Please just keep it as default, thanks.') 
    parser.add_argument('--out', default='attack.png', type=str, help='The output filename for the attacked image.') 
    args = parser.parse_args()

    # load a pretrained VGG16 model on ImageNet 
    # actually I am looking for models pretrained on face dataset instead :((((
    vgg16 = models.vgg16(pretrained=True) # TODO going to update this
    vgg16.features[15].register_forward_hook(capture_feature)

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

    img_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
    ])
    unnormalize = transforms.Normalize(
        mean= [-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std= [1/0.229, 1/0.224, 1/0.255]
    )
    source_img = Image.open(source_img_path).resize((224, 224)).convert("RGB")
    target_img = Image.open(target_img_path).resize((224, 224)).convert("RGB")

    source_img = img_preprocess(source_img).unsqueeze(0)
    target_img = img_preprocess(target_img).unsqueeze(0)

    # to capture the intial features
    vgg16(source_img)
    vgg16(target_img)

    num_iter = args.iter
    eps = args.step_size / 255.0
    max_eps = args.eps / 255.0
    source_feature = intermediate_features[0]
    target_feature = intermediate_features[1]
    adv_img = source_img.clone() + torch.zeros_like(source_img).uniform_(-eps, eps)
    for i in range(num_iter):
        adv_img.requires_grad_()
        zero_gradients(adv_img)

        vgg16(adv_img)
        feature = intermediate_features[-1]

        loss = F.l1_loss(feature, target_feature)
        loss.backward(retain_graph=True)

        adv_img = adv_img.data - eps * torch.sign(adv_img.grad.data)

        adv_img = torch.min(torch.max(adv_img, source_img - max_eps), source_img + max_eps)
        # log progress
        print('Iter: %d / %d. Feature Similarity: %f' % (i+1, num_iter, loss.item()), end='\r')        
    
    outfile = join('output', args.out)
    save_image(unnormalize(adv_img.squeeze(0)), outfile)
    print("Attack Completed. Example is saved as %s" % outfile)
