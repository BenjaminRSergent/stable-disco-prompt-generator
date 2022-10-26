import typing

import PIL
import requests
import torch
import numpy as np 
import utils

_REQ_DIM_FACTOR = 64


def load_img(image_path_or_url: str) -> typing.Tuple[PIL.Image.Image, torch.Tensor]:
    if str(image_path_or_url).startswith("http://") or str(
        image_path_or_url
    ).startswith("https://"):
        image = PIL.Image.open(
            requests.get(image_path_or_url, stream=True, timeout=10).raw
        ).convert("RGB")
    else:
        image = PIL.Image.open(image_path_or_url).convert("RGB")

    return image


def get_size_fixed_larger_dim(image, larger_size=640):
    return scale_size_fixed_larger_dim(*image.size, larger_size)


def scale_size_fixed_larger_dim(orig_width, orig_height, larger_size=640):
    if orig_width > orig_height:
        width = larger_size
        height = utils.round_to_multiple(width * orig_height / orig_width, 64)
    else:
        height = larger_size
        width = utils.round_to_multiple(height * orig_width / orig_height, 64)

    return width, height

def resize_img(img, width, height) -> PIL.Image.Image:
    return img.resize((width, height), resample=PIL.Image.Resampling.BICUBIC)

def make_tiles(img, cols, rows, margin=0):
    img_arr = np.array(img)
    if margin != 0:
        img_arr = img_arr[margin:-margin, margin:-margin] 
    
    width = img_arr.shape[0]//rows
    height = img_arr.shape[1]//cols
     
    
    tiles = [img_arr[x:x+height,y:y+width] for x in range(0,img_arr.shape[0],height) for y in range(0,img_arr.shape[1],width)]
    
    return [PIL.Image.fromarray(tile) for tile in tiles if tile.shape[0] != 0 and tile.shape[1] != 0]