import random

from torch.functional import namedtuple

import ai.stabledisco.constants as sdconsts
import ai.stabledisco.utils.mathutils as mathutils
import ai.torchmodules.utils as torchutils
import clip
import torch

from clip.clip import _tokenizer as clip_tokenizer

WordImpact = namedtuple("WordImpact", "impact word prompt")
TokenImpact = namedtuple("TokenImpact", "impact idx")

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

def rank_word_impact(prompt, clip_model, idxs=None, orig_features=None, normalize=True):
    
    full_features = clip_model.get_features(prompt)[0]
    if orig_features is None:
        orig_features = full_features
    split_prompt = prompt.split()
    
    end_idx = len(split_prompt)-1
    if idxs is None:
        idxs = range(1, end_idx)
    
    orig_diff = 1 - mathutils.cosine_sim( mathutils.norm_t(orig_features),  mathutils.norm_t(full_features)).item()
    impact_tuples = []
    total_impact = 0
    for idx in idxs:
        drop_prompt = ' '.join(split_prompt[:idx] + split_prompt[idx+1:])
        split_features = clip_model.get_features(drop_prompt)[0]
        cos_diff = 1 - (orig_diff +  mathutils.cosine_sim(mathutils.norm_t(orig_features),  mathutils.norm_t(split_features)).item())
        total_impact += cos_diff
        impact_tuples.append([cos_diff, split_prompt[idx], drop_prompt])
        
    impact_tuples.sort(reverse=True)
    
    if normalize:
        for idx in range(len(impact_tuples)):
            impact_tuples[idx][0] /= total_impact
    
    return [WordImpact(*impact) for impact in impact_tuples]

def rank_token_impact(tokens, clip_model, idxs=None, target_features=None, normalize=True):
    full_features = clip_model.get_features(tokens)[0]
    if target_features is None:
        target_features = full_features
    end_idx = find_end_idx(tokens)
    if idxs is None:
        idxs = range(1, end_idx)
        
    if len(idxs) < 2:
        return [TokenImpact(1, 0)]
    
    tokens = tokens.view(-1)
    
    orig_diff = 1 - mathutils.cosine_sim( mathutils.norm_t(target_features),  mathutils.norm_t(full_features)).item()
    impact_tuples = []
    total_impact = 0
    for idx in idxs:
        drop_tokens = torch.cat((tokens[:idx], tokens[idx+1:], torch.zeros(1, device=tokens.device, dtype=tokens.dtype)))
        split_features = clip_model.get_features(drop_tokens)[0]
        cos_diff = 1 - (orig_diff +  mathutils.cosine_sim(mathutils.norm_t(target_features),  mathutils.norm_t(split_features)).item())
        total_impact += cos_diff
        impact_tuples.append([cos_diff, idx])
        
    if normalize:
        for idx in range(len(impact_tuples)):
            impact_tuples[idx][0] /= total_impact
    
    return [TokenImpact(*impact) for impact in impact_tuples]

def trim_prompt(prompt, clip_model, thresh=0.1, orig_features=None):
    if not prompt:
        return
    curr_prompt = prompt
    next_prompt = curr_prompt
    net_impact = 0
    while net_impact < thresh and next_prompt:
        curr_prompt = next_prompt

        last_impact, _, next_prompt = rank_word_impact(curr_prompt, clip_model, orig_features=orig_features)[-1]
        net_impact += last_impact

    return curr_prompt

def decode_tokens(tokens):
    if isinstance(tokens, list):
        tokens = torch.stack(tuple(tokens))
    if len(tokens.shape) == 1:
        tokens = tokens.view(1, *tokens.shape)

    tokens = change_rev(tokens, False).view(tokens.shape)

    texts = [clip_tokenizer.decode(toks.cpu().numpy()) for toks in tokens]
    for idx in range(len(texts)):
        texts[idx] = texts[idx].replace("<|startoftext|>", "")
        end_idx = texts[idx].find("<|endoftext|>")

        if end_idx != -1:
            texts[idx] = texts[idx][:end_idx]

    return texts
