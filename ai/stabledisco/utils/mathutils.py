import ai.stabledisco.constants as sdconsts
import torch


def norm_t(x, dim=-1, keepdim=True):
    return x / x.norm(dim=dim, keepdim=keepdim)

def cosine_sim(x, y):
    return x @ y.T

def calc_singular_vecs(features, cutoff=0.9, largest=True):
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
            ret += svd.S[idx]*svd.Vh[idx]
        else:
            break
        
    return norm_t(ret)


def remove_projection(to_project, axis):
    return to_project - project_to_axis(to_project, axis)
    
def project_to_axis(to_project, axis):
    mag = torch.dot(to_project, axis)/axis.norm(dim=-1, keepdim=True)
    return mag * axis

def make_random_feature_shifts(cnt=1, mean=0.025, std=0.25, min_scale=0.05, device=None, dtype=torch.float):
    shift = make_random_features(cnt, device=device)
    scale = random_scalar_norm(cnt, mean, std, min_scale, device=device, dtype=dtype)
    return scale, (scale * shift).view(-1)

def make_random_features(cnt=1, device=None, dtype=torch.float):
    return norm_t(torch.randn((cnt, sdconsts.feature_width), device=device, dtype=dtype))

def random_scalar_norm(cnt=1, mean=0.025, std=0.25, min_scale=0.05, device=None, dtype=torch.float):
    return torch.abs(torch.randn((cnt, 1),  device=device, dtype=dtype) * std + mean) + min_scale
