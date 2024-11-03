import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info

import models
from models.base import BaseModel
from models.utils import chunk_batch
import renderers
from systems.utils import update_module_step


@models.register('dnerf')
class NeRFModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.renderer = renderers.make(self.config.renderer.name, self.config.renderer)
        if self.config.smpl.optimize_smpl:
            self.smpl_embedding = models.make(self.config.smpl.name, self.config.smpl)
        self.deformer = models.make(self.config.deformer.name, self.config.deformer)

        self.background_color = None
        self.noise = int(self.training)
        self.global_step = 0
        self.current_epoch = 0

    def update_step(self, epoch, global_step):
        self.global_step = global_step
        self.current_epoch = epoch
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        self.noise = int(global_step < 1000 and not self.config.smpl.get("refine", False) and self.training)

    def isosurface(self, **kwargs):
        mesh = self.geometry.isosurface(**kwargs)
        return mesh

    def get_rgb_density(self, points_canonical, valid):
        # for a (usually) small percentatge of positions, the root finding diverges
        rgb_canonical = torch.zeros_like(points_canonical).float()
        density_canonical = -torch.ones_like(points_canonical[..., 0]).float() * 1e5 * int(self.training)
        if valid.any():
            density_canonical[valid], feature = self.geometry(points_canonical[valid])
            rgb_canonical[valid] = self.texture(feature)
            if not self.training:
                density_canonical[valid] = torch.nan_to_num(density_canonical[valid], 0, 0, 0)
                rgb_canonical[valid] = torch.nan_to_num(rgb_canonical[valid], 0, 0, 0)
            else:
                if not torch.isfinite(points_canonical).all():
                    rank_zero_info("WARNING: NaN found in points_canonical")
                if not torch.isfinite(density_canonical).all():
                    rank_zero_info("WARNING: NaN found in density_canonical")
        density_canonical, idx = torch.max(density_canonical, dim=-1)
        rgb_canonical = torch.gather(rgb_canonical, 1, idx[:, None, None].repeat(1, 1, 3))
        return rgb_canonical.reshape(-1, 3), density_canonical.reshape(-1), valid

    def rbg_sigma_fn(self, points):
        points_canonical, valid = self.deformer(points, eval_mode=notself.training)
        rgb, sigma, _ = self.get_rgb_density(points_canonical, valid)
        return rgb, sigma, valid

    def get_density(self, points_canonical, valid):
        density_canonical = -torch.ones_like(points_canonical[..., 0]).float() * 1e5 * int(self.training)
        if valid.any():
            density_canonical[valid], _ = self.geometry(points_canonical[valid])
            if not self.training:
                density_canonical[valid] = torch.nan_to_num(density_canonical[valid], 0, 0, 0)
            else:
                if not torch.isfinite(points_canonical).all():
                    rank_zero_info("WARNING: NaN found in points_canonical")
                if not torch.isfinite(density_canonical).all():
                    rank_zero_info("WARNING: NaN found in density_canonical")
        density_canonical, _ = torch.max(density_canonical, dim=-1)
        return density_canonical.reshape(-1), valid

    def sigma_fn(self, points):
        points_canonical, valid = self.deformer(points)
        return self.get_density(points_canonical, valid)

    def forward_(self, rays_o, rays_d, bg_color):
        rays_o, rays_d, near, far = self.deformer.transform_rays_w2s(rays_o, rays_d)
        # transform_rays_w2s adds a singleton dim
        rays_o, rays_d, near, far = rays_o.squeeze(), rays_d.squeeze(), near.squeeze(), far.squeeze()
        return self.renderer(
            rays_o, rays_d, near, far, self.rbg_sigma_fn, noise=self.noise, bg_color=bg_color,
        )

    @torch.no_grad()
    def render_image(self, batch, img_size):
        if hasattr(self, "smpl_embedding") and self.config.smpl.get("refine", False):
            self.smpl_embedding.prepare_batch(batch)

        self.deformer.prepare_deformer(batch)
        if hasattr(self.renderer, "occupancy_grid_test"):
            self.renderer.occupancy_grid_test.initialize(self.deformer, self.sigma_fn)

        d = self(batch['rays_o'], batch['rays_d'], batch.get("bg_color", None))
        rgb = d["comp_rgb"].reshape(-1, *img_size, 3)
        depth = d["depth"].reshape(-1, *img_size)
        alpha = d["opacity"].reshape(-1, *img_size)
        counter = d["n_samples"].reshape(-1, *img_size)
        return rgb, depth, alpha, counter

    def update_occupancy_grid(self, step):
        if step % 16 == 0 and hasattr(self.renderer, "occupancy_grid_train"):
            _, density, occupied = self.renderer.occupancy_grid_train.update(self.deformer, self.sigma_fn, step)
            reg = 20 * density[~occupied].mean()
            if step < 500:
                reg += 0.5 * density.mean()
            return reg

    def forward(self, rays_o, rays_d, bg_color):
        if self.training:
            out = self.forward_(rays_o, rays_d, bg_color)
        else:
            N, HW = rays_o.shape[:2]
            out = chunk_batch(  # flatten inputs to allow for chunking
                self.forward_,
                self.config.ray_chunk, False,
                rays_o.reshape(N*HW, 3),
                rays_d.reshape(N*HW, 3),
                bg_color.reshape(N*HW, 3) if bg_color is not None else None,
            )
            out["comp_rgb"] = out["comp_rgb"].reshape(N, HW, 3)
            out["depth"] = out["depth"].reshape(N, HW)
            out["opacity"] = out["opacity"].reshape(N, HW)
            out["weights"] = out["weights"].reshape(N, HW, -1)
        return {
            **out,
        }

    def train(self, mode=True):
        return super().train(mode=mode)

    def eval(self):
        return super().eval()

    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank))
            rgb = self.texture(feature).clamp(0,1)
            mesh['v_rgb'] = rgb.cpu()
        return mesh