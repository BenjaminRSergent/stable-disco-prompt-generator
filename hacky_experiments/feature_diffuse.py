import contextlib
import datetime
import importlib
import json
import os
import random
import sys
import time
from collections import defaultdict

print("Hi")
print(os.getenv("PYTHONPATH"))
import ai.stabledisco as sd
import ai.stabledisco.data as sddata
import ai.stabledisco.utils as sdutils
import ai.torchmodules as torchmodules
import ai.torchmodules.data as torchdata
import ai.torchmodules.utils as torchutils
import clip
import ipyplot
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from clip.clip import _tokenizer as clip_tokenizer
from utils import Stopwatch, get_default_path

vit14_clip_model, vit14_clip_preprocess = clip.load('ViT-L/14')
vit14_clip_model = vit14_clip_model.cuda()
clip_model = sd.ClipModel(vit14_clip_model, vit14_clip_preprocess, "name")
sd_model = sdutils.load_default_sd_model()

prompts = ["Testing"]
to_make = 1
curr_seed = 27
width = 768
height = 512
cfg_scale = 9.5
steps = 70
for prompt in prompts:
    for seed in range(to_make):
        curr_seed += 1


        prompt_img = sd.make_prompt(prompt, sd_model,
                                    downscale=8,
                                    width=width, height=height,
                                    scale=cfg_scale, steps=steps,
                                    seed=curr_seed)
        img_width = 512

        ipyplot.plot_images([], max_images=1, img_width=width)
        ipyplot.show()
        print(f"Prompt: {prompt}")
        print(f"Width: {width} | Height: {height} | Seed: {curr_seed} | CFG: {cfg_scale:0.1f} | Steps: {steps}")
