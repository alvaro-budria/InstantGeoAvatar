import os
import torch
import torch.nn.functional as F

import models
from models.base import BaseModel
import renderers
from renderers.renderer_utils import composite

from torch.utils.cpp_extension import load
cuda_dir = os.path.join(os.path.dirname(__file__), "cuda")
raymarch_kernel = load(name='raymarch_kernel',
                       extra_cuda_cflags=[],
                       sources=[f'{cuda_dir}/raymarcher.cpp',
                                f'{cuda_dir}/raymarcher.cu'])


@renderers.register('renderer-instantgeoavatar')
class RendererAcc(BaseModel):
    def setup(self) -> None:
        """
            MAX_SAMPLES_PER_RAY: number of samples per ray
            MAX_NUM_RAYS: max samples to evaluate per batch 
        """
        self.MAX_SAMPLES_PER_RAY = self.config.max_samples_per_ray
        self.MAX_SAMPLES_PER_RAY_TEST = self.config.max_samples_per_ray_test
        self.MAX_NUM_RAYS = self.config.max_num_rays

        self.scene_aabb = torch.tensor(
            [[-1.25, -1.55, -1.25],
             [ 1.25,  0.95,  1.25]]
        ).float().to(self.rank)
        self.occupancy_grid_train = models.make(self.config.occupancy_grid.name, self.config.occupancy_grid)
        self.occupancy_grid_train.scene_aabb = self.scene_aabb
        self.occupancy_grid_test = models.make(self.config.occupancy_grid.name, self.config.occupancy_grid)

    def render(self, rays_o, rays_d, near, far, rbg_sigma_fn, noise, stratified, bg_color):
        rays_o_shape, near_shape = rays_o.shape, near.shape
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        near, far = near.reshape(-1), far.reshape(-1)

        N_step = self.MAX_SAMPLES_PER_RAY if self.training else self.MAX_SAMPLES_PER_RAY_TEST
        step_size = (far - near) / N_step

        occupancy_grid = self.occupancy_grid_train if self.training else self.occupancy_grid_test
        offset = occupancy_grid.min_corner
        scale = occupancy_grid.max_corner - occupancy_grid.min_corner

        z_vals = raymarch_kernel.raymarch_train(
            rays_o, rays_d, near, far,
            occupancy_grid.occupancy_field, scale, offset,
            step_size, N_step,
        )
        mask = z_vals > 0

        if stratified:
            z_vals = z_vals + torch.rand_like(z_vals) * step_size[:, None]
        pts = z_vals[..., None] * rays_d[:, None] + rays_o[:, None]

        rbg_out = torch.zeros_like(pts, dtype=torch.float32)
        sigma_out = -torch.ones_like(rbg_out[..., 0], dtype=torch.float32) * 1e3
        sdg_grad = torch.ones_like(rbg_out, dtype=torch.float32)
        valid_samples = torch.zeros_like(pts[..., 0], dtype=torch.float32)
        if mask.sum() > 0:
            rbg_out[mask], sigma_out[mask], sdg_grad[mask], valid_samples[mask] = rbg_sigma_fn(pts[mask])
        if noise > 0:
            sigma_out = sigma_out + noise * torch.randn_like(sigma_out)

        dists = torch.ones_like(sigma_out) * step_size[:, None]
        weights, transmittance = composite(sigma_out.reshape(z_vals.shape), dists, thresh=0)
        no_hit = transmittance[..., -1]

        color = (weights[..., None] * rbg_out.reshape(pts.shape)).sum(dim=-2)
        if bg_color is not None:
            bg_color = bg_color.reshape(-1, 3)
            color = color + no_hit[..., None] * bg_color
        else:
            color = color + no_hit[..., None]
        depth = (weights * z_vals).sum(dim=-1)

        normals = F.normalize(sdg_grad, p=2, dim=-1)
        normal_starts, normal_ends = normals[:-1], normals[1:]
        normal_mids = (normal_starts + normal_ends) / 2.

        ray_normals = (weights[...,None] * normals).sum(dim=-2)
        ray_normals = F.normalize(ray_normals, p=2, dim=-1)

        return {
            "comp_rgb": color.reshape(rays_o_shape),
            "depth": depth.reshape(near_shape),
            "opacity": (weights.sum(-1)).reshape(near_shape),
            "weights": weights.reshape(*near_shape, -1),
            "n_samples": torch.zeros_like(depth),
            "sdfgrad": sdg_grad,
            "normals": (normal_starts, normal_mids, normal_ends),
            "ray_normals": ray_normals,
            "valid_samples": valid_samples,
        }

    def forward(self, rays_o, rays_d, near, far, rbg_sigma_fn, noise=0, stratified=False, bg_color=None):
        if not self.training:
            with torch.no_grad():
                return self.render(rays_o, rays_d, near, far, rbg_sigma_fn, 0, False, bg_color)
        else:
            return self.render(rays_o, rays_d, near, far, rbg_sigma_fn, noise, True, bg_color)
