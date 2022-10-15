import random

import ai.stabledisco.constants as sdconsts
import ai.torchmodules.utils as torchutils
import clip
import torch


def random_prompt_combo(prompts, length=77, device=None):
    tokens_set = set(sum([clip.tokenize(prompt, truncate=True)[0].tolist() for prompt in prompts], []))

    tokens_set.remove(sdconsts.sot_token)
    tokens_set.remove(sdconsts.eot_token)
    tokens_set.remove(0)
    tokens_lst = list(tokens_set)

    prompt_tokens = [0 for _ in range(sdconsts.prompt_token_len)]
    prompt_tokens[0] = sdconsts.sot_token
    prompt_tokens[length-1] = sdconsts.eot_token
    for idx in range(1, length):
        prompt_tokens[idx] = random.choice(tokens_lst)

    if device is None:
        device = torchutils.get_default_device()
    prompt_tokens = torch.tensor(prompt_tokens, device=device)

    return prompt_tokens

def find_end_idx(tokens):
    end_idx_arg = torch.argwhere(tokens == sdconsts.eot_token).view(-1)
    if end_idx_arg.size(0) == 0:
        return -1
    return end_idx_arg[0]
    
def get_single_word_token(word):
    tokenized = clip.tokenize(word)[0]
    if tokenized[2].item() != sdconsts.eot_token:
        raise Exception(f"Input {word} consists of multiple tokens {tokenized}")
    return tokenized[1].item()

def rev_tokens(text_tokens):
    if len(text_tokens.shape) > 1:
        text_tokens.view(1, text_tokens.size(0))
    flipped = torch.flip(text_tokens, dims=(1,))
    end_idx_arg = torch.argwhere(flipped == sdconsts.eot_token)[:,1]
    for idx in range(flipped.size(0)):
        flipped[idx] = torch.cat((flipped[idx,end_idx_arg[idx]:], flipped[idx,:end_idx_arg[idx]]))
    return flipped