from torch import einsum
import torch.nn.functional as F
from pytorch3d import ops


def skinning_mask(x, w, bone_transf):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        bone_transf (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    p, n = w.shape
    w_tf = einsum("pn,nij->pij", w, bone_transf.squeeze(0))
    x_h = x_h.view(p, 1, 4).expand(p, 4, 4)
    x_h = (w_tf * x_h).sum(-1)
    return x_h[:, :3]


def bmv(m, v):
    return (m * v.transpose(-1, -2).expand(-1, 3, -1)).sum(-1, keepdim=True)


def query_weights_smpl(x, smpl_verts, smpl_weights, resolution=128):
    # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
    dist, idx, _ = ops.knn_points(x, smpl_verts.detach(), K=1)
    dist = dist.sqrt().clamp_(0.0001, 1.)
    weights = smpl_weights[0, idx]

    ws = 1. / dist
    ws = ws / ws.sum(-1, keepdim=True)
    weights = (ws[..., None] * weights).sum(-2)

    b, c, d, h, w = 1, 24, resolution // 4, resolution, resolution
    weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w)
    for _ in range(2):
        mean=(weights[:,:,2:,1:-1,1:-1]+weights[:,:,:-2,1:-1,1:-1]+\
              weights[:,:,1:-1,2:,1:-1]+weights[:,:,1:-1,:-2,1:-1]+\
              weights[:,:,1:-1,1:-1,2:]+weights[:,:,1:-1,1:-1,:-2])/6.0
        weights[:, :, 1:-1, 1:-1, 1:-1] = (weights[:, :, 1:-1, 1:-1, 1:-1] - mean) * 0.7 + mean
        sums = weights.sum(1, keepdim=True)
        weights = weights / sums
    return weights.detach()