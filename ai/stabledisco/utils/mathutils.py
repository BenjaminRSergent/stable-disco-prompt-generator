import torch


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
        
    return ret / ret.norm(dim=-1, keepdim=True)


def remove_projection(to_project, axis):
    return to_project - project_to_axis(to_project, axis)
    

def project_to_axis(to_project, axis):
    mag = torch.dot(to_project, axis)/axis.norm(dim=-1, keepdim=True)
    return mag * axis
