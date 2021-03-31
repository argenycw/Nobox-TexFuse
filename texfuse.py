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
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd.gradcheck import zero_gradients

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceFuse by AG')
    parser.add_argument('--source', default='', type=str, help='the path of the source image')
    parser.add_argument('--target', default='', type=str, help='the path of the target image') 
    parser.add_argument('--eps', default=50, type=int, help='Maximum perturbation strength (over 255)') 
    parser.add_argument('--iter', default=50, type=int, help='Number of iterations for projected gradient computation') 
    parser.add_argument('--step_size', default=5, type=int, help='Please just keep it as default, thanks.') 
    parser.add_argument('-o', '--out', default='attack.png', type=str, help='The output filename for the attacked image.') 
    parser.add_argument('-u', '--upsample', action='store_true', help='Upsample the perturbation rather than doublesample the output.')
    parser.add_argument('-n', '--noresize', action='store_true', help='Prohibit the script from resizing the images (also accept rectanuglar shape)') 
    args = parser.parse_args()

    # load a pretrained VGG16 model on ImageNet 
    # actually I am looking for models pretrained on face dataset instead :((((
    vgg16 = models.vgg16(pretrained=True) # TODO going to update this
    vgg16 = vgg16_manipulate(vgg16)
    #vgg16.features[15].register_forward_hook(capture_feature)

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

    source_raw = Image.open(source_img_path).convert("RGB")
    target_raw = Image.open(target_img_path).convert("RGB")
    source_w, source_h = source_raw.size
    target_w, target_h = target_raw.size    

    if abs(1 - ((source_w / target_w) / (source_h / target_h))) > 0.1:
        print('Warning: A huge difference between dimensions in source and target is detected. Note the image may be stretched to a certain degree.')
    if args.noresize:
        if source_w != target_w or source_h != target_h:
            print('Resizing target (%d, %d) into source (%d, %d).' % (target_w, target_h, source_w, source_h))            
        source_img = source_raw
        target_img = target_raw.resize((source_w, source_h))
    else:
        source_img = source_raw.resize((224, 224))
        target_img = target_raw.resize((224, 224))

    source_img = img_preprocess(source_img).unsqueeze(0)
    target_img = img_preprocess(target_img).unsqueeze(0)

    # to capture the intial features
    source_feature = vgg16(source_img)
    target_feature = vgg16(target_img)

    num_iter = args.iter
    eps = args.step_size / 255.0
    max_eps = args.eps / 255.0
    #source_feature = intermediate_features[0]
    #target_feature = intermediate_features[1]
    adv_img = source_img.clone() + torch.zeros_like(source_img).uniform_(-eps, eps)
    for i in range(num_iter):
        adv_img.requires_grad_()
        zero_gradients(adv_img)

        feature = vgg16(adv_img)
        #feature = intermediate_features[-1]

        loss = F.l1_loss(feature, target_feature)
        loss.backward(retain_graph=True)

        adv_img = adv_img.data - eps * torch.sign(adv_img.grad.data)

        adv_img = torch.min(torch.max(adv_img, source_img - max_eps), source_img + max_eps)
        # log progress
        print('Iter: %d / %d. Feature Similarity: %f' % (i+1, num_iter, loss.item()), end='\r')        
    
    # upsample the perturbation when set
    if args.upsample:
        delta = adv_img - source_img
        delta_large = F.interpolate(delta, size=(source_h, source_w), mode='bicubic', align_corners=False)
        adv_img = img_preprocess(source_raw) + delta_large

    outfile = join('output', args.out)
    save_image(unnormalize(adv_img.squeeze(0)), outfile)
    print("Attack Completed. Example is saved as %s" % outfile)
