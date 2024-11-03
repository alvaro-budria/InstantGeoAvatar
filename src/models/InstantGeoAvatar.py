import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_info

import models
from models.DNeRF import NeRFModel
from models.network_utils import get_embedder_Hann


@models.register('instantgeoavatar')
class VolSDFModel(NeRFModel):
    def beta_fn(self):
        return self.geometry.get_beta()

    def grad_deformed(self, x_c, x_d, dsdf_dxc, create_graph, retain_graph):
        # X_c: N, 3
        # X_d: N, 3
        # dsdf_dxc: N, 3
        d_out = torch.zeros_like(x_c, requires_grad=False).unsqueeze(0).expand(3, -1, -1)  # 3', N, 3
        for i in range(d_out.shape[0]):
            d_out[i, :, i] = 1
        grad = torch.autograd.grad(
            outputs=x_c,
            inputs=x_d,
            grad_outputs=d_out,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
            is_grads_batched=True,
        )[0]
        grad = grad.permute(1, 0, 2)  # N, 3', 3
        n_d = torch.einsum('bi,bij->bj', dsdf_dxc, grad)
        n_d = torch.nn.functional.normalize(n_d, dim=1)
        return n_d

    def get_rgb_density(self, points_canonical, valid):
        # for a (usually) small percentatge of positions, the root finding diverges
        rgb_canonical = torch.zeros_like(points_canonical).float()
        density_canonical = -torch.ones_like(points_canonical[..., 0]).float() * 1e5 * int(self.training)
        sdf_grad = torch.ones_like(points_canonical).float()
        if valid.any():
            sdf_canonical, sdf_grad[valid], feature = self.geometry(points_canonical[valid], with_grad=True, with_feature=True)
            rgb_canonical[valid] = self.texture(feature, normals=F.normalize(sdf_grad[valid], dim=-1))
            if not self.training:
                sdf_canonical = torch.nan_to_num(sdf_canonical, 0, 0, 0)
                rgb_canonical[valid] = torch.nan_to_num(rgb_canonical[valid], 0, 0, 0)
            else:
                if not torch.isfinite(points_canonical).all():
                    rank_zero_info("WARNING: NaN found in points_canonical")
                if not torch.isfinite(sdf_canonical).all():
                    rank_zero_info("WARNING: NaN found in sdf_canonical")
            density_canonical[valid] = self.geometry.get_density(sdf_canonical)
        density_canonical, idx = torch.max(density_canonical, dim=-1)
        rgb_canonical = torch.gather(rgb_canonical, 1, idx[:, None, None].repeat(1, 1, 3))
        sdf_grad = torch.gather(sdf_grad, 1, idx[:, None, None].repeat(1, 1, 3))
        valid = torch.gather(valid, 1, idx[:, None].repeat(1, 1))
        return rgb_canonical.reshape(-1, 3), density_canonical.reshape(-1), sdf_grad.reshape(-1, 3), valid.reshape(-1)

    def deform_points(self, points, eval_mode):
        points_canonical, valid = self.deformer(points, eval_mode=eval_mode)

        kick_in_epoch = 1
        deformed = False
        if valid.any() and self.current_epoch >= kick_in_epoch and hasattr(self, "non_rigid_motion"):
            # add delta to canonical points
            embed_fn, embed_dim = get_embedder_Hann(
                multires=6, epoch_val=self.current_epoch,
                kick_in_epoch=kick_in_epoch, full_band_epoch=kick_in_epoch+5,
            )
            embedded_points = embed_fn(points_canonical[valid])

            pose_cond = self.deformer.smpl_params["body_pose"]

            pose_cond = pose_cond.expand((embedded_points.shape[0], pose_cond.shape[-1]))
            points_canonical = points_canonical.clone()
            points_canonical[valid] = self.non_rigid_motion(
                embedded_points, points_canonical[valid], pose_cond,
            )
            deformed = True
        return points_canonical, valid, deformed

    def rbg_sigma_fn(self, points):
        points_canonical, valid, deformed = self.deform_points(points, eval_mode=True)

        rgb, sigma, sdf_grad, valid = self.get_rgb_density(points_canonical, valid)

        if self.training and not deformed and hasattr(self, "non_rigid_motion"):  # add weights multiplied by 0 to rgb, to accomodate for PyTorch's DDP
            rgb = rgb + 0 * sum([p.sum() for p in self.non_rigid_motion.parameters() if p.requires_grad])

        return rgb, sigma, sdf_grad, valid.float()

    def get_sdf_density(self, points_canonical, valid, beta=None):
        sdf_canonical = -torch.ones_like(points_canonical[..., 0]).float() * 1e5 * int(self.training)
        density_canonical = sdf_canonical.clone()
        if valid.any():
            sdf_canonical[valid] = self.geometry(points_canonical[valid], with_grad=False, with_feature=False)
            if not self.training:
                sdf_canonical[valid] = torch.nan_to_num(sdf_canonical[valid], 0, 0, 0)
            else:
                if not torch.isfinite(points_canonical).all():
                    rank_zero_info("WARNING: NaN found in points_canonical")
                if not torch.isfinite(sdf_canonical).all():
                    rank_zero_info("WARNING: NaN found in sdf_canonical")
            density_canonical[valid] = self.geometry.get_density(sdf_canonical[valid], beta=beta[valid] if beta is not None else beta)
        density_canonical, idx = torch.max(density_canonical, dim=-1)
        sdf_canonical = torch.gather(sdf_canonical, 1, idx[:, None])
        return sdf_canonical.reshape(-1), density_canonical.reshape(-1), valid

    def sdf_fn_canonical(self, points_canonical):
        return self.geometry(points_canonical, with_grad=True, with_feature=False)

    def sigma_from_points_fn(self, points, eval_mode, beta=None):
        points_canonical, valid = self.deformer(points, eval_mode=True)
        beta_expanded = beta.unsqueeze(-1).repeat(1, 13) if beta is not None else None
        return self.get_sdf_density(points_canonical, valid, beta_expanded)[1:3]

    def update_occupancy_grid(self, step):
        if step % 16 == 0 and hasattr(self.renderer, "occupancy_grid_train"):
            _, density, occupied = self.renderer.occupancy_grid_train.update(
                                        self.deformer, self.sigma_from_points_fn, step
                                   )
            reg = 20 * density[~occupied].mean()
            if step < 500:
                reg += 0.5 * density.mean()
            return reg

    @torch.no_grad()
    def render_image(self, batch, img_size):
        if hasattr(self, "smpl_embedding") and self.config.smpl.get("refine", False):
            self.smpl_embedding.prepare_batch(batch)

        self.deformer.prepare_deformer(batch, delta_weights_predictor=self.delta_weights_predictor if hasattr(self, "delta_weights_predictor") else None)

        with torch.inference_mode(False):  # scale and offset inside geometry must not be inference tensors
            self.geometry.initialize(self.deformer.scene_aabb, force_reinit=True)

        if hasattr(self.renderer, "occupancy_grid_test"):
            self.renderer.occupancy_grid_test.initialize(self.deformer, self.sigma_from_points_fn)

        d = self(batch['rays_o'], batch['rays_d'], batch.get("bg_color", None))
        rgb = d["comp_rgb"].reshape(-1, *img_size, 3)
        depth = d["depth"].reshape(-1, *img_size)
        alpha = d["opacity"].reshape(-1, *img_size)
        ray_normals = d["ray_normals"].reshape(-1, *img_size, 3)
        counter = d["n_samples"].reshape(-1, *img_size)
        return rgb, depth, alpha, ray_normals, counter

    def forward_(self, rays_o, rays_d, bg_color):
        rays_o, rays_d, near, far = self.deformer.transform_rays_w2s(rays_o, rays_d)
        # transform_rays_w2s adds a singleton dim
        rays_o, rays_d, near, far = rays_o.squeeze(), rays_d.squeeze(), near.squeeze(), far.squeeze()
        return self.renderer(
            rays_o, rays_d,
            near, far,
            self.rbg_sigma_fn,
            noise=self.noise,
            bg_color=bg_color,
        )

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface(
            deformer=self.deform_points, get_sdf_density=self.get_sdf_density,
        ) if export_config.get('deform_isosurface', True) else self.isosurface()
        return mesh

    def grad_canonical(self, pnts_c, sdf, create_graph, retain_graph):
        with torch.inference_mode(False):
            with torch.set_grad_enabled(True):
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                dsdf_dxc = torch.autograd.grad(
                    outputs=sdf,
                    inputs=pnts_c,
                    grad_outputs=d_output,
                    create_graph=create_graph,
                    retain_graph=retain_graph,
                    only_inputs=True,
                    allow_unused=False)[0]
        return dsdf_dxc