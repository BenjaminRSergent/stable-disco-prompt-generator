from contextlib import nullcontext

import torch
from ai.stabledisco.stablediscomodel import StableDiscoModel
from pytorch_lightning import seed_everything
from torch import autocast


def make_img_prompt(
    prompts,
    image,
    model: StableDiscoModel,
    strength=0.8,
    scale=7.5,
    steps=100,
    ddim_eta=0.0,
    iters=1,
    batch_size=1,
    seed=27,
    precision="autocast",
):
    if type(prompts) == str:
        prompts = [prompts]

    seed_everything(seed)

    sampler = model.create_sampler(False)
    data = sum([[batch_size * [prompt] for prompt in prompts]], [])

    precision_scope = autocast if precision == "autocast" else nullcontext

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False)

    init_latent = model.get_image_init_latent(image, batch_size)

    assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(strength * steps)
    print(f"target t_enc is {t_enc} steps with strength {strength}")
    ret = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for _ in range(iters):
                    for prompt in data:
                        uc = None
                        if scale != 1.0:
                            uc = model.get_unconditional_conditioning(batch_size)
                        c = model.get_learned_conditioning(prompt)
                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(
                            init_latent,
                            torch.tensor([t_enc] * batch_size, device=device),
                        )
                        # decode it
                        samples = sampler.decode(
                            z_enc,
                            c,
                            t_enc,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                        )

                        ret += model.reconstruct_ddim(samples)

    return ret


def make_prompt(
    prompts,
    model,
    width=512,
    height=512,
    latient_channels=4,
    scale=7.5,
    steps=100,
    ddim_eta=0.0,
    plms=False,
    iters=1,
    downscale=8,
    batch_size=1,
    seed=27,
    precision="autocast",
):
    if type(prompts) == str:
        prompts = [prompts]

    
    sampler = model.create_sampler(plms)

    data = sum([[batch_size * [prompt] for prompt in prompts]], [])
    seed_everything(seed)
    ret = []
    inter_ret = []
    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for _ in range(iters):
                    for prompt in data:
                        uc = None
                        if scale != 1.0:
                            uc = model.get_unconditional_conditioning(batch_size)
                        c = model.get_learned_conditioning(prompt)
                        

                        shape = [
                            latient_channels,
                            height // downscale,
                            width // downscale,
                        ]

                        samples_ddim, inter = sampler.sample(
                            S=steps,
                            conditioning=c,
                            batch_size=batch_size,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=ddim_eta,
                        )

                        ret += model.reconstruct_ddim(samples_ddim)
                        for x in inter["pred_x0"]:
                            inter_ret += model.reconstruct_ddim(x)

    return ret, inter_ret
