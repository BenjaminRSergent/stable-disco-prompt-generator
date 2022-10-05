import random

import ai.stabledisco.constants as torchconstants
import ai.torchmodules.utils as torchutils
import clip
import torch


def random_prompt_combo(prompts, length=77, device=None):
    tokens_set = set(sum([clip.tokenize(prompt, truncate=True)[0].tolist() for prompt in prompts], []))

    tokens_set.remove(torchconstants.sot_token)
    tokens_set.remove(torchconstants.eot_token)
    tokens_set.remove(0)
    tokens_lst = list(tokens_set)

    prompt_tokens = [0 for _ in range(torchconstants.prompt_token_len)]
    prompt_tokens[0] = torchconstants.sot_token
    prompt_tokens[length-1] = torchconstants.eot_token
    for idx in range(1, length):
        prompt_tokens[idx] = random.choice(tokens_lst)

    if device is None:
        device = torchutils.get_default_device()
    prompt_tokens = torch.tensor(prompt_tokens, device=device)

    return prompt_tokens
