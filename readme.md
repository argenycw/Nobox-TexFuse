# TexFuse 

This is an early version of TexFuse, a no-box adversarial attack generater for deep convolution neural networks. It works by integrating the latent feature of the `target` image into `source` image, possbily fooling a CNN model to recognize `source` as `target`.

## Installation
This program runs in Python3 and requires the following pip modules:
- numpy==1.18.4
- torch==1.7.1
- torchvision==0.4.1
- Pillow==8.0.1

## Models

In current version, the script supports two models pretrained on VGGFace2.
- VGG16 (Default)  `--model vgg` 
- ResNet50 `--model resnet`

You can download the .pth file [here](https://drive.google.com/file/d/1fuNhlJ2X36zXqh6Ab8A253Xe6ztDJZ86/view?usp=sharing). Unzip the files in `pretrained/` folder so that the pretrained weights are stored as `pretrained/resnet50_scratch_dag.pth` and `pretrained/vgg_face_dag.pth` respectively. **Make sure you have the two weights downloaded properly before running the script.**

Since the pretrained weights of the two models are different, it is normal to see different similarity scores with different pretrained models.

## Usage

Example Usage: 
```
python texfuse.py --source <source image> --target <target image>
```

The output image will be saved as `outputs/attack.png` by default. You can overwrite the path with the `-o` or `--out` argument.

The script automatically resizes the picture into dimension 224x224. To disable such feature, you can consider upsampling the perturbation with the argument `-u` or `--upsample`. With `-u` enabled, the perturbation will be upsampled, forming a more apparent "noise" on top of the source image.

```
python texfuse.py --source <source image> --target <target image> --out <output image> -u
```


Batch computation and GPU are currently not supported.
