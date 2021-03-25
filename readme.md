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

The output image will be saved as `outputs/attack.png` by default. You can overwrite the path with the `-o` or `--out` argument.

The script automatically resizes the picture into dimension 224x224. To disable such feature, use the `--noresize` argument:
```
python texfuse.py --source <source image> --target <target image> --out <output image> --noresize
```
Alternatively, you can also consider upsampling the perturbation with the argument `-u` or `--upsample`.

Batch computation and GPU are currently not supported.
