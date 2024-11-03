import torch


def stratified_sampling(N, step_size):
    device = step_size.device
    z = torch.arange(N, device=device) * step_size[..., None]
    z += torch.rand_like(z) * step_size[..., None]
    return z


def composite(sigma_vals, dists, thresh=0):
    # 0 (transparent) <= alpha <= 1 (opaque)
    tau = torch.relu(sigma_vals) * dists
    alpha = 1.0 - torch.exp(-tau)
    if thresh > 0:
        alpha[alpha < thresh] = 0
    transmittance = torch.cat([torch.ones_like(alpha[..., 0:1]),
                               torch.cumprod(1 - alpha + 1e-10, dim=-1)], dim=-1)
    w = alpha * transmittance[..., :-1]
    return w, transmittance


def ray_aabb(o, d, bbox_min, bbox_max):
    t1 = (bbox_min - o) / d
    t2 = (bbox_max - o) / d

    t_min = torch.minimum(t1, t2)
    t_max = torch.maximum(t1, t2)

    near = t_min.max(dim=-1).values
    far = t_max.min(dim=-1).values
    return near, far
