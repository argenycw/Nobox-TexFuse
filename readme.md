# TexFuse 

This is an early version of TexFuse, a no-box adversarial attack generater for deep convolution neural networks. It works by integrating the latent feature of the `target` image into `source` image, possbily fooling a CNN model to recognize `source` as `target`.

## Installation
This program runs in Python3 and requires the following pip modules:
- numpy==1.18.4
- torch==1.7.1
- torchvision==0.4.1
- Pillow==8.0.1


## Usage

Example Usage: 
```
python texfuse.py --source <source image> --target <target image>
```

Batch computation and GPU is currently not supported.
