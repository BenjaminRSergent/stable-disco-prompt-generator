import os

import clip
import numpy as np
import PIL
import torch
from ai.stabledisco.stablediscomodel import StableDiscoModel
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


def load_sd_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()

    return StableDiscoModel(model)


def load_default_sd_model(sd_dir=None):
    if sd_dir is None:
        sd_dir = "/home/ubuntu/development/stablediffusion"
    sd_config = "configs/stable-diffusion/v2-inference-v.yaml"
    config_path = os.path.join(sd_dir, sd_config)

    ckpt = "models/ldm/stable-diffusion-v2/model.ckpt"

    return load_sd_model(config_path, ckpt)


def load_sd_model(sd_config, ckpt, sd_dir=None):
    if sd_dir is None:
        sd_dir = "/home/ubuntu/development/stablediffusion"
    # sd_config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    sd_config = "configs/stable-diffusion/v2-inference-v.yaml"
    config = OmegaConf.load(os.path.join(sd_dir, sd_config))
    return load_sd_model_from_config(
        config,
        os.path.join(sd_dir, ckpt),
    )


def load_clip_model(model_name="ViT-L/14"):
    from ai.stabledisco.clipmodel import ClipModel

    ViTL14_model, preprocess = clip.load(model_name)
    return ClipModel(ViTL14_model, preprocess, model_name)


def transform_img(image):
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0
