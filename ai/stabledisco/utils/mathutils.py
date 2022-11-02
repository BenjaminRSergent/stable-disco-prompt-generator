import ai.stabledisco.constants as sdconsts
import ai.torchmodules.utils as torchutils
import torch
import random

def diff_t(vals):
    return vals[1:] - vals[:-1]

def norm_scalars(x, min_val, max_val):
    return (x - min_val)/(max_val-min_val)

def ease_out_quat(x):
    return -(x * (x - 2))

def ease_in_quat(x):
    return x * x

def ease_in_quat(x):
    return x * x

def ease_in_lin(x):
    return x

def ease_out_lin(x):
    return 1-x

def normed_sine(x):
    return torch.sin(torch.pi/2*x)

def norm_t(x, dim=-1, keepdim=True):
    return x / x.norm(dim=dim, keepdim=keepdim)

def cosine_sim(x, y):
    return unflatten(x) @ unflatten(y).T

def calc_singular_vecs(features, cutoff=0.9, largest=True, weights=None, normalize=True):
    features /= features.norm(dim=-1, keepdim=True)
    svd = torch.linalg.svd(features.float())
    if largest:
        ret = svd.Vh[0]
        loop_range = range(1, svd.S.size(0))
    else:
        ret = svd.Vh[-1]
        loop_range = range(svd.S.size(0)-1, 0, -1)
    print(svd.S)

    for idx in loop_range:
        if (largest and svd.S[idx] > cutoff) or (not largest and svd.S[idx] < cutoff):
            if weights is None:
                weight = 1.0
            elif len(weights)-1 < idx:
                weight = weights[-1]
            else:
                weight = weights[idx]
                
            ret += weight*svd.S[idx]*svd.Vh[idx]
        else:
            break
    if normalize:
        ret = norm_t(ret)
        
    return ret

def remove_projection(to_project, axis):
    return (to_project - project_to_axis(to_project, axis))
    
def project_to_axis(to_project, axis):
    to_project = to_project
    axis = flatten(axis)
    mag = torch.dot(to_project, axis)/axis.norm(dim=-1, keepdim=True)
    return mag * axis

def make_random_feature_shifts(cnt=1, mean=0.025, std=0.25, min_scale=0.05, device=None, dtype=torch.float):
    shift = make_random_features_uniform(cnt, device=device)
    scale = random_scalar_norm(cnt, mean, std, min_scale, device=device, dtype=dtype)
    return (scale * shift).view(-1)

def make_random_features_uniform(cnt=1, device=None, dtype=torch.float):
    if device is None:
        device = torchutils.get_default_device()
    return norm_t(2*(torch.rand((cnt, sdconsts.feature_width), device=device, dtype=dtype) - 0.5))

def make_random_text_features(cnt=1, device=None, dtype=torch.float, normalize=True):
    features = make_random_features_norm(cnt=cnt, device=device, dtype=dtype)
    mean, std = get_text_feature_stats(device=device)
    ret = features * std + mean
    if normalize:
        ret = norm_t(ret)
    
    return ret

def make_random_features_norm(cnt=1, device=None, dtype=torch.float):
    if device is None:
        device = torchutils.get_default_device()

    return norm_t(torch.randn((cnt, sdconsts.feature_width), device=device, dtype=dtype))

def random_scalar_norm(cnt=1, mean=0.025, std=0.25, min_scale=0.05, device=None, dtype=torch.float):
    return torch.abs(torch.randn((cnt, 1),  device=device, dtype=dtype) * std + mean) + min_scale

def flatten(tens):
    return tens.view(-1)

def unflatten(tens):
    if len(tens.shape) == 1:
        tens = tens.view(1, -1)
    return tens
