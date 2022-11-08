import typing

import einops
import numpy as np
import PIL
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class StableDiscoModel:
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def reconstruct_ddim(self, samples_ddim: torch.tensor) -> typing.List[PIL.Image.Image]:
        ret = []
        x_samples_ddim = self._model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * einops.rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            ret.append(PIL.Image.fromarray(x_sample.astype(np.uint8)))
        return ret

    def create_sampler(self, use_plms: bool) -> typing.Union[PLMSSampler, DDIMSampler]:
        if use_plms:
            return PLMSSampler(self._model)

        return DDIMSampler(self._model)

    def ema_scope(self):
        return self._model.ema_scope()

    def get_image_init_latent(self, image: PIL.Image.Image, batch_size=1) -> torch.Tensor:
        image = self.preprocess_image(image).to(self._model.device)
        init_image = einops.repeat(image, "1 ... -> b ...", b=batch_size)
        return self._model.get_first_stage_encoding(self._model.encode_first_stage(init_image))  # move to latent space

    def get_learned_conditioning(self, text: str) -> torch.Tensor:
        return self._model.get_learned_conditioning(text)

    def get_unconditional_conditioning(self, batch_size=1) -> torch.Tensor:
        return self.get_learned_conditioning(batch_size * [""])

    @staticmethod
    def preprocess_image(image: PIL.Image.Image) -> torch.Tensor:
        w, h = image.size
        # resize to integer multiple of 32
        w, h = map(lambda x: x - x % 32, (w, h))
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0
