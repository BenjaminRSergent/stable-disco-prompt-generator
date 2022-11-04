import random
from torch.cuda import traceback

from torch.functional import namedtuple
import ai.stabledisco.utils.imageutils as imgutils
import ai.stabledisco.constants as sdconsts
import ai.stabledisco.utils.mathutils as mathutils
import ai.torchmodules.utils as torchutils
import clip
import torch

from clip.clip import _tokenizer as clip_tokenizer

import utils

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
        return torch.tensor([-1], device=tokens.device)
    return end_idx_arg.long()
    
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
    # Add spaces between punctuation and numbers
    prompt = decode_tokens(clip.tokenize(prompt)[0])[0]
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
    if isinstance(prompt, torch.Tensor):
        prompt = decode_tokens(prompt)
    if not prompt:
        return
    curr_prompt = prompt
    next_prompt = curr_prompt
    net_impact = 0
    while net_impact < thresh and next_prompt:
        curr_prompt = next_prompt

        last_impact = rank_word_impact(curr_prompt, clip_model, orig_features=orig_features)
        if not last_impact:
            return ""
        last_impact, _, next_prompt = last_impact[-1]
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

def improve_url_features_sum(urls_to_use, clip_model, feature_improver, display_func=None, **kwargs):
    return improve_feature_sum(get_img_features(urls_to_use, clip_model, display_func=display_func)[0], feature_improver, **kwargs)
    
def calc_improved_prompt_features(base_prompts, clip_model, feature_improver, **kwargs):
    features = tuple((clip_model.encode_text(prompt, truncate=True) for prompt in base_prompts))
    feature_stack = torch.stack(features).view((len(base_prompts), -1))
    return improve_feature_sum(feature_stack, feature_improver, **kwargs)

def improve_feature_sum(features_list, feature_improver, weights=None,
                        target_rating=8.5, target_prob=0.98, max_prob_diff=0.15, max_rating_diff=0.01,
                        use_sing=False, sing_cuttoff=1.0):
    if weights is None:
        weights = []
    
    if isinstance(features_list, torch.Tensor) and len(features_list.shape) == 0:
        features_list = [features_list]
        
    if use_sing:
        features_list = [mathutils.calc_singular_vecs(features_list, sing_cuttoff)]
        
    all_features = [feature_improver.improve_features(features,
                                     target_rating=target_rating, max_prob_diff=max_prob_diff,
                                     target_prob=target_prob, max_rating_diff=max_rating_diff) for features in features_list]

    target_features = torch.zeros(all_features[0].shape, device=all_features[0].device, dtype=all_features[0].dtype)
    for idx, features in enumerate(all_features[1:]):
        weight = 1
        if idx < len(weights):
            weight = weights[idx]
        target_features = add_features(target_features, features, weight)
        
    target_features = mathutils.norm_t(target_features)

    return feature_improver.improve_features(target_features,
                                             target_rating=target_rating, max_prob_diff=max_prob_diff,
                                             target_prob=target_prob, max_rating_diff=max_rating_diff)

def get_features_and_singular(urls, clip_model, cutoff=0.75, display_func=None, largest=True, cuda=True, normalize=True):
    all_features, _ = get_img_features(urls,clip_model,  display_func=display_func, cuda=cuda, normalize=normalize)
    return all_features, mathutils.calc_singular_vecs(all_features, cutoff=cutoff, largest=largest)

def get_img_features(urls, clip_model, display_func=None, cuda=True, normalize=True):
    _, all_features, _, _ = get_imgs_and_features(urls, clip_model, display_func=display_func, cuda=cuda, normalize=normalize)
    return all_features, mathutils.norm_t(all_features.sum(axis=0))

def get_imgs_and_features(urls, clip_model, larger_width=704, display_func=None, cuda=True, normalize=True):
    if isinstance(urls, str) or not hasattr(urls, '__iter__'):
        urls = [urls]
        
    all_features = []
    all_imgs = []
    total_width = 0
    total_height = 0
    for url in urls:
        try:
            torchutils.torch_garbage_collect()
            img = imgutils.load_img(url)
            if display_func:
                display_func(img)
            width, height = imgutils.get_size_fixed_larger_dim(img, larger_width)
            total_width += width
            total_height += height

            features = clip_model.encode_image_features(img)[0].float()
            
            if normalize:
                features /= features.norm(dim=-1, keepdim=True)
                
            all_features.append(features)
            all_imgs.append(img.resize((width, height)))
            
        except Exception as ex:
            print(f"Failed to load {url}")
            print(traceback.format_exc(ex))
            
    
    gen_width = int(utils.round_to_multiple(total_width/len(urls), 64))
    gen_height = int(utils.round_to_multiple(total_height/len(urls), 64))
    
    all_features = torch.stack(tuple(all_features))
    if not cuda:
        all_features = all_features.cpu()
    
    return all_imgs, all_features, gen_width, gen_height

def blend_text_features(features, other_features, alpha=1.0):
    text_means, text_stds = get_text_feature_stats(features.device)
    ret = features(1-alpha) + other_features*alpha
    
    zero_centered = ret - text_means
    fix_variance  = zero_centered/text_stds
    return mathutils.norm_t(fix_variance + text_means)

def add_text_features(features, other_features, scale=1.0):
    text_means, text_stds = get_text_feature_stats(features.device)
    ret = features + other_features*scale
    scaled_text_means = text_means*scale
    zero_centered = ret - scaled_text_means
    fix_variance  = zero_centered/text_stds
    return mathutils.norm_t(fix_variance + scaled_text_means)

def add_features(features, other_features, scale=1.0):
    _, text_stds = get_text_feature_stats(features.device)
    avg_mean = torch.mean(torch.stack((features, other_features*scale)), dim=0)
    return (features + other_features*scale - avg_mean)/text_stds + avg_mean

def make_random_text_features(cnt=1, device=None, dtype=torch.float, normalize=True):
    features = mathutils.make_random_features_norm(cnt=cnt, device=device, dtype=dtype)
    mean, std = get_text_feature_stats(device=device)
    ret = features * std + mean
    if normalize:
        ret = mathutils.norm_t(ret)
    
    return ret


def get_text_feature_stats(device = None):
    if device is None:
        device = torchutils.get_default_device()
    
    if device not in _text_feature_stats_dict:
        _text_feature_stats_dict[device] = (_text_feature_means.to(device), _text_feature_stds.to(device))
    
    return _text_feature_stats_dict[device]
    

_text_feature_stats_dict = {}

_text_feature_means = torch.Tensor([-3.9030e-03,  2.7220e-02,  2.1757e-02, -9.2000e-02, -5.6538e-02,
         1.0373e-01, -2.5380e-02, -1.3268e-02,  8.0159e-03,  3.7787e-02,
         1.6964e-01, -6.8325e-02,  7.9276e-02,  6.0124e-02, -8.1522e-02,
        -1.0647e-01, -5.3842e-02,  1.2506e-01,  1.3543e-01,  6.9913e-02,
         5.9512e-02,  1.9998e-02, -2.1073e-02,  1.0911e-01,  2.5636e-01,
        -2.8690e-02, -6.4374e-02, -7.5167e-02,  1.3921e-02, -1.6335e-01,
         2.8250e-03, -3.5808e-02, -5.5518e-03,  1.5763e-01,  9.3999e-02,
         4.1578e-03, -6.7069e-02, -5.0322e-02,  3.7059e-02, -4.4961e-02,
         1.5649e-02,  6.9082e-02, -1.1327e-02, -8.3307e-02,  4.0316e-02,
         4.6049e-03,  1.7241e-02, -8.6442e-02,  5.2078e-02,  6.3878e-02,
         1.0353e-01, -3.7198e-03, -2.4376e-02,  8.9394e-02,  8.7039e-02,
        -8.0625e-02,  1.7610e-01,  7.2331e-03,  7.2671e-03, -1.0427e-01,
         6.9047e-02,  2.0354e-03, -1.5379e-01,  9.9312e-02, -4.3297e-02,
         3.2534e-02,  1.1183e-01, -1.4419e-01,  3.5907e-02,  3.2965e-02,
         1.0091e-01, -1.3386e-01,  3.6704e-02,  1.0017e-01, -4.1287e-02,
        -1.0110e-01,  3.6971e-02,  3.0244e-02,  5.2239e-02,  2.6405e-02,
        -8.1690e-02, -1.2964e-02, -7.7783e-02, -3.0898e-02,  1.4186e-01,
        -2.4680e-02, -6.5456e-02,  5.0611e-02,  4.6669e-03, -1.7452e-01,
         1.1564e-01, -3.4528e-02, -1.2609e-02,  3.0613e-03, -9.6083e-02,
         3.1268e-02, -1.8010e-02, -3.1503e-02,  9.9689e-02, -5.1675e-02,
        -3.1000e-03,  1.2372e-02,  7.3941e-02, -7.5393e-02,  6.7700e-02,
        -1.7290e-02,  6.3483e-02, -9.0542e-02, -5.1114e-02, -1.0735e-01,
        -5.0420e-02,  7.4444e-02, -6.3832e-03,  2.0048e-02, -3.4173e-02,
         8.0200e-03, -1.5965e-01,  1.5473e-02,  1.1721e-01,  8.0945e-02,
        -4.1346e-02, -8.3093e-02, -4.5878e-02,  7.2064e-02,  4.0572e-02,
         1.5977e-01, -7.3023e-02,  9.7854e-03,  6.0444e-02,  6.8957e-02,
         2.3657e-03,  4.1622e-02, -4.8168e-02, -1.4134e-01,  5.2478e-02,
        -3.4727e-01,  7.1498e-02,  4.9867e-02,  1.0443e-01,  1.4713e-01,
         3.4546e-02,  1.7145e-01,  6.7913e-02,  1.1333e-01,  3.6573e-02,
        -1.3326e-01, -2.3407e-02,  3.6143e-01, -2.2869e-02, -2.1041e-02,
         9.7822e-03, -2.7263e-02,  6.5220e-02, -1.3527e-01, -3.4195e-02,
        -6.9999e-02, -1.9364e-01, -9.2739e-02,  2.4070e-02, -5.9854e-02,
        -4.2213e-02, -4.5577e-02, -5.0762e-02,  1.1429e-02,  2.3136e-03,
        -9.5726e-02, -1.6797e-01, -5.5357e-02,  3.8138e-03,  1.2639e-02,
        -1.7878e-02,  4.8069e-02, -7.8390e-02, -2.3080e-02,  1.5699e-01,
         1.6627e-01, -8.9491e-02, -1.8615e-01,  2.8276e-01, -4.3971e-02,
        -8.4095e-02, -3.2272e-02,  4.5869e-02,  2.5922e-02, -1.8224e-02,
        -1.9488e-02,  5.4214e-02, -3.8047e-03,  3.8738e-02, -1.7778e-01,
        -3.4557e-02,  1.2082e-02,  4.1124e-02,  9.4857e-02,  1.7144e-01,
         5.2192e-02,  3.9493e-02, -1.2828e-02,  2.1438e-02,  1.2899e-01,
         2.8845e-02, -5.1408e-01,  1.7309e-01,  5.1029e-02, -4.6200e-02,
        -4.4909e-02, -6.4826e-03,  3.1131e-02,  9.4044e-03, -7.4734e-02,
         1.7796e-02, -2.1049e-02, -3.6745e-02,  3.2119e-02, -9.3961e-02,
         1.0474e-01, -4.0345e-02,  1.7961e-01, -6.9780e-02,  1.2886e-02,
         1.0039e-02,  1.4051e-01,  5.7264e-02, -3.7458e-02, -7.6370e-02,
         6.3450e-02, -9.0174e-02, -5.1251e-02, -4.4047e-03, -7.3730e-02,
         1.0811e-01, -8.5298e-02, -2.3278e-01,  1.4283e-01,  2.2865e-02,
        -7.7548e-02, -6.2268e-02,  3.4289e-02, -2.5469e-02,  5.8454e-02,
         6.3064e-02, -1.4756e-01, -6.9516e-02, -8.1063e-02, -1.1241e-02,
         2.6658e-01, -9.7791e-02,  1.6748e-02,  3.8810e-02, -3.8152e-02,
         2.5418e-02,  7.9797e-02,  1.6224e-01,  8.2757e-02,  9.0381e-02,
         5.0938e-02, -9.6723e-02, -1.0874e-01, -6.5026e-03,  8.3721e-03,
         1.4436e-02,  9.6765e-02, -3.2198e-02,  2.6233e-02,  2.3205e-02,
        -2.5055e-01,  1.2428e-02,  3.4067e-02, -5.3269e-04,  1.2659e-02,
        -6.2914e-03,  5.4442e-02,  9.2348e-02, -4.4553e-02, -1.0585e-01,
        -1.6865e-02, -6.2539e-02,  3.0831e-02, -6.5043e-02, -2.0146e-01,
         2.3035e-02, -3.5365e-01,  4.6027e-02, -1.2940e-03, -2.0001e-02,
         2.4457e-02,  3.8150e-02,  4.1214e-02,  2.0391e-01,  7.4570e-03,
         4.0373e-02,  3.8748e-02, -6.8321e-02, -1.0099e-01, -6.3266e-02,
        -4.4827e-02, -7.8120e-02, -1.3465e-01, -1.0543e-02,  8.1918e-03,
        -4.5415e-02, -6.8485e-02, -7.1599e-04, -1.3331e-01,  1.8294e-02,
        -8.3643e-02, -1.6622e-01,  1.9991e-02,  3.4921e-02, -8.9075e-02,
         2.3071e-01,  7.6159e-02, -2.2732e-01,  1.2114e-01, -7.7197e-02,
        -1.5228e-04, -3.3228e+00, -1.6238e-01, -1.7263e-03, -5.7848e-02,
         7.7076e-02, -6.1252e-02,  9.2957e-02, -1.0113e-01, -4.0404e-02,
        -5.7482e-02, -2.1480e-01,  2.2399e-02,  3.8349e-02,  4.8120e-02,
        -2.7790e-01,  7.9586e-02, -2.7974e-02, -8.6389e-02, -2.8092e-03,
        -5.9092e-02,  3.5501e-02,  4.1311e-02,  7.5680e-02,  3.8417e-02,
         4.1646e-02, -8.3955e-02, -2.0499e-02, -7.4059e-02,  7.6211e-02,
         9.3274e-03, -2.3842e-02,  4.2653e-02,  8.3683e-02, -1.1446e-01,
        -5.5059e-02, -3.0898e-02,  1.3188e-02, -1.6277e-01,  2.8930e-02,
         1.3823e-01,  1.2799e-01, -5.9577e-03, -5.3694e-02, -1.4055e-02,
        -8.4215e-02, -3.8107e-02,  2.2933e-01,  2.8481e-01,  7.6664e-03,
        -5.4292e-02,  1.8282e-02, -1.6133e-01,  4.8487e-02, -3.5588e-02,
        -4.5800e-02,  2.1450e-02,  8.4332e-02,  1.1821e-01, -9.6300e-02,
        -1.9037e-02, -2.5261e-02,  5.8775e-02,  6.4175e-02,  2.4536e-02,
         4.6501e-02, -1.2253e-02,  7.2440e-02, -2.5817e-02, -6.0550e-02,
        -1.2980e-01,  1.3310e-02, -2.1400e-01,  6.8351e-02,  5.6582e-02,
         9.0652e-02, -5.1605e-02,  5.8550e-02, -8.3935e-02,  7.3856e-02,
         1.5515e-01,  6.8468e-02, -1.7711e-01,  5.4158e-02, -6.8067e-02,
        -1.6893e-01,  7.4500e-02,  1.1866e-01,  3.4797e-02, -6.5038e-01,
         5.9130e-02, -9.4850e-02, -6.5983e-02, -7.1184e-02, -3.1624e-02,
        -3.6434e-02, -6.0906e-03,  3.4664e-02, -3.6732e-01,  8.1498e-03,
         7.5432e-02,  7.4640e-02,  1.0052e-01,  4.8077e-02, -1.4596e-02,
        -1.7264e-01,  7.5755e-02, -9.8514e-03,  3.7913e-03,  5.5944e-02,
         1.4872e-01, -9.1648e-02, -8.1611e-03,  5.1629e-02,  1.2481e-01,
        -9.0870e-02, -3.6576e-02, -7.4949e-02, -9.2607e-02, -9.0554e-02,
         2.5399e-01,  9.4787e-03, -5.0437e-02,  3.1257e-02, -4.0913e-02,
         3.0294e+00, -5.1230e-02, -2.7469e-02,  2.3193e-02, -3.6305e-02,
         1.9020e-02,  1.5958e-01, -8.9452e-02,  5.7925e-02,  5.7043e-02,
        -3.9042e-02, -5.1926e-02, -7.0640e-02,  3.2813e-02, -1.2496e-01,
        -1.1969e-02, -3.3036e-02, -4.1557e-02,  3.4573e-02, -2.0947e-01,
        -5.0238e-02, -3.3656e-02, -1.5689e-02,  3.3220e-01, -2.9326e-01,
         5.7039e-02,  1.0328e-01, -1.5910e-01,  7.7722e-02, -4.1889e-02,
         1.2989e-02, -1.2860e-01, -1.4957e-01, -3.3467e-02, -3.6439e-02,
         2.8400e-04,  4.3582e-02, -3.3862e-01, -5.7401e-03, -4.0951e-02,
         2.7690e-02,  1.6002e-01,  1.9047e-02, -1.0967e-02,  1.9783e-02,
        -5.9742e-02, -1.0602e-01, -7.2992e-02,  6.4953e-02, -1.9424e-01,
        -6.9879e-02, -1.2961e-01, -2.9446e-01,  1.1141e-01,  1.3981e-01,
         1.0795e-01, -6.8056e-02,  1.3491e-01,  6.2816e-02, -2.9691e-02,
        -3.3366e-02,  3.5077e-02,  1.8529e-02,  3.5317e-02, -3.0050e-02,
        -1.3930e-01, -9.4620e-02,  6.7216e-04, -1.3715e-01,  5.6877e-02,
        -6.1626e-02, -7.7888e-03, -2.3599e-01,  5.0594e-02,  5.0709e-03,
        -3.9390e-03,  1.1459e-01, -5.9877e-03, -1.8062e-02, -9.7970e-02,
         1.1828e-01,  5.1914e-03, -9.0407e-02,  3.9359e-02, -6.8518e-02,
        -2.1240e-01, -5.8492e-02, -9.2463e-04, -2.1616e-01, -2.9277e-03,
         8.1714e-02,  1.5247e-01,  1.0267e-01,  1.0104e-01,  6.7933e-03,
        -5.9796e-02,  3.4311e-02,  4.9859e-02, -7.0678e-03, -3.8818e-02,
         5.0864e-02,  7.2852e-02, -3.8684e-02,  7.1414e-02,  1.1136e-02,
        -2.3702e-02,  2.0412e-02,  5.1056e-03,  5.2242e-02, -6.0740e-03,
        -7.0913e-02, -1.8734e-01,  2.1858e-02, -7.6176e-02,  3.8694e-02,
         1.1106e-02, -1.4813e-01,  1.9714e-02, -1.3598e-01,  5.4129e-02,
         2.0698e-01, -4.3128e-02,  9.8701e-03,  6.9949e-02,  1.1639e-02,
         1.6748e-02, -9.8118e-02, -1.4283e-03,  1.2661e-01,  7.6396e-02,
         6.4171e-02, -6.8862e-02,  1.0719e-01,  3.6415e-02,  1.7260e-02,
        -6.7135e-02,  8.9366e-02,  7.5665e-02, -1.4185e-02,  4.4169e-02,
        -4.3261e-02,  8.5846e-02,  3.4782e-02, -5.0153e-03,  5.3239e-02,
         5.4949e-02, -1.0355e-01, -9.8937e-02,  3.6799e-01, -2.6107e-02,
        -4.2862e-02,  2.4766e-02,  7.0329e-02, -5.1178e-02,  1.2108e-02,
         4.3414e-02,  3.5691e-02, -1.0766e-01,  1.8453e-02, -1.4365e-02,
        -3.6368e-02, -4.7613e-02,  8.3647e-02,  4.0079e-02,  1.4135e-03,
        -8.2367e-01,  3.1262e-02,  2.1143e-04,  1.3946e-01, -1.2678e-02,
        -9.2160e-02, -1.9643e-02,  5.2482e-02,  2.3720e-04,  3.5564e-02,
        -5.9787e-02, -4.7258e-02, -1.0020e-01,  8.8103e-03, -6.0619e-02,
         4.7315e-02, -5.2004e-02, -4.0700e-02,  7.3535e-03, -1.7306e-01,
         1.6443e-01, -1.2637e-02, -3.9662e-02, -9.3355e-02, -9.3789e-02,
        -2.4907e-02, -6.2203e-02,  4.7940e-02,  1.2586e-01, -9.7767e-02,
        -6.5507e-02, -6.4302e-02,  1.3106e-01,  5.0944e-02,  4.5281e-02,
         8.6916e-02,  8.3872e-03,  1.8608e-01,  7.5743e-02,  3.7350e-01,
        -6.5189e-02,  1.3682e-01,  7.0999e-04,  5.7992e-02, -1.2424e-01,
        -7.0563e-02,  1.5500e-01,  3.1839e-02, -1.5328e-02,  9.2034e-02,
         3.7037e-02,  5.9731e-02, -1.1306e-01, -3.3426e-02,  3.6622e-01,
         3.1310e-03,  4.2805e-02, -8.7610e-02,  1.0158e-02,  1.9809e-02,
        -9.7799e-02,  3.4239e-02,  1.2565e-01, -1.1074e-01,  1.4302e-01,
        -3.9035e-02,  3.2089e-02,  6.6692e-02, -7.1521e-03, -1.1534e-01,
        -1.2728e-02, -1.1202e-01,  3.7731e-02,  2.2622e-02, -8.1097e-02,
        -3.8326e-02, -8.4873e-02,  2.8240e-02, -5.3095e-02, -2.6315e-02,
         1.6674e-02, -9.4798e-03,  6.0555e-02,  1.2451e-01, -2.0515e-02,
         3.2703e-02, -4.2488e-02,  7.9939e-01, -9.2029e-02, -2.0901e-02,
         1.4726e-01, -6.3853e-02, -1.1519e-02, -5.7297e-02, -5.6016e-03,
        -9.9261e-02, -8.6626e-02, -7.9309e-02, -2.4565e-01,  1.0062e-01,
         8.2404e-03,  1.7117e-01, -5.2004e-02,  9.1925e-02, -5.0515e-02,
         4.9307e-02, -1.2473e-02, -9.0763e-02,  5.4920e-01,  1.7497e-02,
         6.8150e-02,  6.3537e-02, -4.9837e-02,  3.2235e-02, -7.3325e-02,
        -1.3644e-01, -9.3907e-03, -1.2608e-03, -1.9031e-01,  3.9437e-01,
        -2.1833e-01,  1.5020e-01,  1.8118e-02, -8.9871e-02,  3.2903e-02,
         4.9406e-02, -7.2388e-02, -6.6577e-02,  3.7793e-03, -1.5864e-01,
         5.0976e-03,  1.8288e-02,  3.0157e-02, -2.9366e-02, -6.2253e-02,
         4.9896e-02, -4.1694e-02,  1.9458e-02, -1.3391e-01, -2.0768e-02,
         1.5172e-01,  1.2607e-01,  9.1943e-04, -4.1568e-02, -1.3925e-01,
        -2.7244e-02, -5.9337e-02,  5.0016e-02, -3.9249e-02,  5.9922e-02,
        -4.2586e-01, -1.1503e-01,  1.0299e-01, -5.1732e-02, -4.1523e-02,
         1.4723e-01, -1.4296e-03, -1.0973e-01,  1.0078e-01,  3.5652e-02,
         3.0485e-02, -1.0123e-01, -2.8446e-02])
_text_feature_stds = torch.Tensor([0.4071, 0.4321, 0.3445, 0.5552, 0.3659, 0.3634, 0.3865, 0.3544, 0.3825,
        0.3908, 0.3877, 0.3740, 0.3601, 0.3803, 0.4031, 0.3726, 0.4541, 0.3711,
        0.4022, 0.3698, 0.4076, 0.3657, 0.3834, 0.3808, 0.4945, 0.3683, 0.3723,
        0.3824, 0.4430, 0.3843, 0.3733, 0.3781, 0.3750, 0.3818, 0.3742, 0.3708,
        0.3948, 0.3492, 0.3696, 0.3879, 0.3914, 0.3566, 0.3928, 0.3377, 0.3627,
        0.3646, 0.3716, 0.3918, 0.3552, 0.4289, 0.4027, 0.3809, 0.3818, 0.3963,
        0.3523, 0.3866, 0.3903, 0.3884, 0.3980, 0.4113, 0.3834, 0.3988, 0.3786,
        0.3633, 0.3882, 0.3920, 0.3944, 0.3820, 0.3810, 0.3773, 0.3823, 0.3864,
        0.3812, 0.3500, 0.3788, 0.3624, 0.4002, 0.4021, 0.3976, 0.3810, 0.4118,
        0.3643, 0.3969, 0.3677, 0.3928, 0.3625, 0.3930, 0.4044, 0.4333, 0.3579,
        0.3985, 0.3659, 0.3718, 0.4027, 0.3701, 0.3600, 0.3625, 0.3676, 0.3645,
        0.3539, 0.3806, 0.3869, 0.3714, 0.3723, 0.3693, 0.4076, 0.3652, 0.3545,
        0.3785, 0.4277, 0.3747, 0.3994, 0.3717, 0.3808, 0.3773, 0.3494, 0.4236,
        0.3686, 0.5593, 0.4177, 0.3614, 0.3590, 0.3872, 0.3533, 0.3416, 0.4564,
        0.3727, 0.3885, 0.3723, 0.3854, 0.3733, 0.4183, 0.3695, 0.4161, 0.3646,
        0.6748, 0.3970, 0.3810, 0.3881, 0.3873, 0.3741, 0.3902, 0.3801, 0.3957,
        0.4215, 0.3813, 0.3843, 0.5771, 0.3970, 0.3783, 0.3783, 0.3609, 0.3547,
        0.3829, 0.3842, 0.3938, 0.3741, 0.4268, 0.3729, 0.3507, 0.3703, 0.3831,
        0.3548, 0.3929, 0.3879, 0.3789, 0.3832, 0.3470, 0.3758, 0.3800, 0.3970,
        0.3890, 0.3716, 0.3814, 0.3906, 0.4446, 0.4073, 0.4433, 0.5095, 0.3945,
        0.6125, 0.3703, 0.3754, 0.3905, 0.3890, 0.3700, 0.4118, 0.3809, 0.3693,
        0.4101, 0.3625, 0.3980, 0.4060, 0.3859, 0.4010, 0.8624, 0.4263, 0.3930,
        0.3699, 0.3889, 0.3672, 0.8963, 0.3839, 0.3749, 0.3961, 0.3431, 0.3536,
        0.3649, 0.3881, 0.3935, 0.3721, 0.3638, 0.4066, 0.3629, 0.3814, 0.3768,
        0.4605, 0.4037, 0.4010, 0.4623, 0.4025, 0.4475, 0.3707, 0.4022, 0.3840,
        0.3765, 0.3856, 0.4089, 0.3746, 0.3721, 0.3750, 0.3925, 0.4715, 0.4947,
        0.4016, 0.3716, 0.3841, 0.4014, 0.3781, 0.3676, 0.3902, 0.3759, 0.3815,
        0.3843, 0.3821, 0.4342, 0.3763, 0.4433, 0.3704, 0.4132, 0.3709, 0.3723,
        0.3927, 0.4226, 0.3926, 0.3872, 0.3968, 0.4281, 0.3688, 0.6220, 0.3712,
        0.3864, 0.3709, 0.3758, 0.3650, 0.4074, 0.3716, 0.3822, 0.3674, 0.3759,
        0.3718, 0.4149, 0.3829, 0.3766, 0.3576, 0.3573, 0.3687, 0.3727, 0.3678,
        0.4149, 0.3741, 0.5811, 0.3790, 0.4934, 0.4022, 0.3829, 0.3662, 0.3823,
        0.9032, 0.3753, 0.3823, 0.3693, 0.3909, 0.3595, 0.3513, 0.3566, 0.4446,
        0.4034, 0.3806, 0.3659, 0.3702, 0.3941, 0.3847, 0.3871, 0.3773, 0.3660,
        0.3787, 0.4026, 0.4097, 0.3737, 0.4597, 0.3994, 0.4036, 0.4572, 0.3777,
        0.3663, 2.9047, 0.4049, 0.3590, 0.4176, 0.3802, 0.3877, 0.3827, 0.3475,
        0.3850, 0.3895, 0.5514, 0.4074, 0.3896, 0.5955, 0.4235, 0.3617, 0.3912,
        0.3757, 0.3776, 0.3795, 0.3819, 0.3672, 0.3643, 0.4286, 0.3617, 0.3958,
        0.3748, 0.4213, 0.4058, 0.3798, 0.3983, 0.4423, 0.3666, 0.3531, 0.3757,
        0.3829, 0.3823, 0.3749, 0.3866, 0.3817, 0.3758, 0.4034, 0.3965, 0.3889,
        0.4000, 0.3818, 0.4798, 0.7440, 0.3798, 0.3685, 0.5056, 0.4289, 0.3742,
        0.3769, 0.3777, 0.3624, 0.3548, 0.3830, 0.3772, 0.3826, 0.3703, 0.3764,
        0.3856, 0.3487, 0.4492, 0.3790, 0.3742, 0.3696, 0.3982, 0.4915, 0.3715,
        0.3773, 0.4221, 0.3761, 0.4543, 0.4014, 0.3686, 0.3820, 0.3743, 0.3792,
        0.3824, 0.4360, 0.3888, 0.3920, 0.4135, 0.3667, 0.3745, 0.4284, 0.5562,
        0.3866, 0.3805, 0.3714, 0.3944, 0.3794, 0.3818, 0.3885, 0.4123, 0.3363,
        0.3647, 0.3983, 0.4243, 0.3770, 0.3681, 0.3721, 0.6275, 0.3664, 0.3871,
        0.3914, 0.3625, 0.3780, 0.3770, 0.3756, 0.3586, 0.3731, 0.3732, 0.3806,
        0.3951, 0.3876, 0.3766, 0.4848, 0.3750, 0.3820, 0.3880, 0.3684, 2.6080,
        0.3717, 0.3902, 0.3677, 0.3799, 0.3835, 0.3843, 0.4133, 0.3840, 0.3763,
        0.6243, 0.3855, 0.5682, 0.3675, 0.4028, 0.3888, 0.3585, 0.3910, 0.3765,
        0.3857, 0.3871, 0.3829, 0.3792, 0.4799, 0.5563, 0.3774, 0.3751, 0.7293,
        0.3772, 0.3591, 0.3675, 0.3819, 0.3967, 0.3692, 0.3725, 0.3965, 0.3779,
        0.4535, 0.3738, 0.3657, 0.3901, 0.3581, 0.3851, 0.3533, 0.3785, 0.3612,
        0.3738, 0.3690, 0.4070, 0.3814, 0.3757, 0.3844, 0.4493, 0.3664, 0.3798,
        0.3815, 0.3842, 0.3893, 0.3993, 0.3773, 0.3833, 0.3830, 0.4033, 0.3756,
        0.4133, 0.3706, 0.3656, 0.3711, 0.5309, 0.3750, 0.3977, 0.3725, 0.4825,
        0.3740, 0.3864, 0.3742, 0.3554, 0.4164, 0.4003, 0.5494, 0.3838, 0.3832,
        0.3767, 0.3570, 0.3761, 0.4397, 0.3512, 0.3796, 0.5572, 0.3590, 0.3919,
        0.4084, 0.3482, 0.4195, 0.3778, 0.3576, 0.3604, 0.3904, 0.4584, 0.3807,
        0.4481, 0.3762, 0.4110, 0.3683, 0.4045, 0.3834, 0.3729, 0.3599, 0.4805,
        0.3740, 0.3883, 0.4317, 0.3628, 0.3763, 0.3963, 0.3972, 0.6222, 0.3805,
        0.3575, 0.3834, 0.5446, 0.3539, 0.3719, 0.3868, 0.5129, 0.3561, 0.3845,
        0.3553, 0.4023, 0.3757, 0.3880, 0.4334, 0.3878, 0.3663, 0.3783, 0.3954,
        0.3832, 0.3377, 0.4047, 0.3774, 0.3976, 0.3665, 0.3925, 0.3882, 0.3817,
        0.3797, 0.3718, 0.3907, 0.5796, 0.3801, 0.3857, 0.3883, 0.4109, 0.4052,
        0.3802, 0.3663, 0.3760, 0.3992, 0.3551, 0.3659, 0.4014, 0.4108, 0.3647,
        0.3742, 0.3787, 0.5587, 0.3719, 0.3780, 0.4430, 0.3870, 0.3565, 0.3882,
        0.3697, 0.3890, 0.3861, 0.3934, 0.3905, 0.4651, 0.4055, 0.3704, 0.3680,
        0.3773, 0.3773, 0.3842, 0.4984, 0.5989, 0.4376, 0.3857, 0.3682, 0.4030,
        0.4055, 0.3739, 0.3651, 0.4248, 0.3819, 0.5681, 0.3739, 0.5576, 0.3702,
        0.4016, 0.3744, 0.3955, 0.3683, 0.3860, 0.3361, 0.4276, 0.3965, 0.3697,
        0.3709, 0.3873, 0.3792, 0.4448, 0.3793, 0.3718, 0.4261, 0.3649, 0.3856,
        0.3556, 0.3894, 0.4599, 0.3720, 0.3622, 0.4105, 0.3715, 0.3647, 0.3707,
        0.3932, 0.3607, 0.3933, 0.5831, 0.3892, 0.3663, 0.3240, 0.3959, 0.3817,
        0.3973, 0.4097, 0.3677, 0.3849, 0.4028, 0.3856, 0.3703, 0.3652, 0.3891,
        0.3865, 0.3692, 0.3632, 0.3746, 0.4121, 0.3695, 0.3707, 0.3709, 0.5587,
        0.3676, 0.3561, 0.3809, 0.4213, 0.3925, 0.3755, 0.3545, 0.3870, 0.3657,
        0.3896, 0.6202, 0.3765, 0.3635, 0.3649, 0.3972, 0.3632, 0.3608, 0.3585,
        0.3620, 0.3684, 0.4850, 0.4149, 0.3729, 0.3792, 0.3782, 0.3764, 0.3819,
        0.3855, 0.3875, 0.4067, 0.6144, 0.5575, 0.5424, 0.3979, 0.3664, 0.3840,
        0.3671, 0.3731, 0.3856, 0.3778, 0.3695, 0.3997, 0.3980, 0.3926, 0.3726,
        0.3656, 0.3781, 0.3738, 0.3867, 0.3720, 0.3650, 0.3673, 0.3964, 0.3716,
        0.3493, 0.3671, 0.3691, 0.3683, 0.3607, 0.3671, 0.3766, 0.3685, 0.4906,
        0.3767, 0.3940, 0.3664, 0.3942, 0.3794, 0.3673, 0.4055, 0.4438, 0.3509,
        0.3619, 0.3794, 0.3732])