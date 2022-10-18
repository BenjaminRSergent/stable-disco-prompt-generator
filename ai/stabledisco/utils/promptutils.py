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
    if isinstance(tokens, torch.Tensor) and len(tokens.shape) == 1:
        tokens = tokens.unsqueeze(0)
    end_token = sdconsts.sot_token if is_rev_tokens(tokens) else sdconsts.eot_token
    end_idx_arg = torch.argwhere(tokens == end_token)[:,1]
    if end_idx_arg.size(0) == 0:
        return -1
    return end_idx_arg
    
def get_single_word_token(word):
    tokenized = clip.tokenize(word)[0]
    if tokenized[2].item() != sdconsts.eot_token:
        raise Exception(f"Input {word} consists of multiple tokens {tokenized}")
    return tokenized[1].item()

def change_rev(text_tokens, ret_rev):
    if isinstance(text_tokens, torch.Tensor) and len(text_tokens.shape) == 1:
        text_tokens = text_tokens.unsqueeze(0)
        
    if is_rev_tokens(text_tokens) != ret_rev:
        text_tokens = rev_tokens(text_tokens)
        
    return text_tokens

def rev_tokens(text_tokens):
    if isinstance(text_tokens, torch.Tensor) and len(text_tokens.shape) == 1:
        text_tokens = text_tokens.unsqueeze(0)
    
    flipped = torch.flip(text_tokens, dims=(1,))
    search_token = sdconsts.sot_token if is_rev_tokens(text_tokens) else sdconsts.eot_token
    end_idx_arg = torch.argwhere(flipped == search_token)[:,1]
    
    for flipped_idx, token_idx in enumerate(end_idx_arg):
        flipped[flipped_idx] = torch.cat((flipped[flipped_idx,token_idx:], flipped[flipped_idx,:token_idx]))
    return flipped.long()

def is_rev_tokens(text_tokens):
    if isinstance(text_tokens, list):
        return text_tokens[0][0] == sdconsts.eot_token
    return text_tokens.view(-1)[0] == sdconsts.eot_token

