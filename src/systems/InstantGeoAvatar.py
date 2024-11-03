import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

import torch
import torch.nn.functional as F

import losses
import models
import systems
from systems.base import BaseSystem
from models.utils import scale_anything
from models.network_utils import get_embedder_Hann
from models.network_utils import NonRigidMotionMLP_TCNN


class PointInSpace:
    def __init__(self, global_sigma=0.5, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input=None, local_sigma=None, global_ratio=0.125):
        """Sample one point near each of the given point + 1/8 uniformly.
        Args:
            pc_input (tensor): sampling centers. shape: [B, D]
        Returns:
            samples (tensor): sampled points. shape: [B + B / 8, D]
        """
        batch_size, dim = pc_input.shape
        if local_sigma is None:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma)
        sample_global = (
            torch.rand(int(batch_size * global_ratio), dim, device=pc_input.device)
            * (self.global_sigma * 2)
        ) - self.global_sigma
        sample = torch.cat([sample_local, sample_global], dim=0)

        return sample


@systems.register('instantgeoavatar-system')
class InstantGeoAvatarSystem(BaseSystem):
    def prepare(self, datamodule):
        self.loss_fn = losses.make(self.config.system.loss.name, self.config.system.loss)
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.datamodule = datamodule
        self.model = models.make(self.config.model.name, self.config.model)
        if self.config.model.smpl.optimize_smpl:
            self.model.smpl_embedding.fill_parameters(self.datamodule.trainset.get_SMPL_params())

        n_frames = len(self.datamodule.trainset)
        print('n_frames: ', n_frames)
        embedderHann, out_embed_size = get_embedder_Hann(
            6, epoch_val=self.current_epoch, kick_in_epoch=20, full_band_epoch=40
        )
        self.model.non_rigid_motion = NonRigidMotionMLP_TCNN(
            pos_embed_size=out_embed_size, condition_code_size=69, mlp_width=128, mlp_depth=5, skips=[4]
        )

        self.sampler = PointInSpace()

    def forward(self, batch):
        return self.model(batch['rays_o'], batch['rays_d'], batch.get("bg_color", None))

    def preprocess_data(self, batch, stage):
        if hasattr(self.model, "smpl_embedding"):
            batch = self.model.smpl_embedding.prepare_batch(
                batch, substitute=self.current_epoch >= self.config.model.smpl.start
            )

    def training_step(self, batch, *args, **kwargs):
        batch["current_epoch"] = self.current_epoch
        self.model.timestep = batch["timestep"]
        self.model.geometry.current_epoch = self.current_epoch
        self.model.deformer.current_epoch = self.current_epoch

        if self.current_epoch >= 2 and hasattr(self.model, "pose_corrector"):
            batch["body_pose_before"] = batch["body_pose"].clone()
            batch["is_pose_corrected"] = True
            batch["body_pose"] = self.model.pose_corrector(batch["body_pose"], batch["idx"])

        self.model.deformer.prepare_deformer(batch, delta_weights_predictor=self.model.delta_weights_predictor if hasattr(self.model, "delta_weights_predictor") else None)
        reg_density = self.model.update_occupancy_grid(self.global_step)
        self.model.geometry.initialize(self.model.deformer.scene_aabb)
        if hasattr(self.model, "smpl_embedding"):
            for k in ["body_pose", "global_orient", "transl"]:
                gt = self.datamodule.trainset.smpl_params[k]
                gt = torch.from_numpy(gt).cuda()
                self.log(f"train/{k}", F.l1_loss(getattr(self.model.smpl_embedding, k).weight, gt), prog_bar=True)

        if isinstance(self.model.deformer, type("SMPLDeformer")):
            raise ValueError("SMPLDeformer not supported")

        rendered = self(batch)

        losses = self.loss_fn(rendered, batch, self.global_step, self.current_epoch)

        if not (reg_density is None or self.model.config.smpl.refine_smpl):
            losses["reg_density"] = reg_density
            losses["loss"] += reg_density * self.C(self.config.system.loss.lambda_density)

        for k, v in losses.items():
            self.log(f"train/{k}", v)

        if 'inv_s' in rendered:
            self.log('train/inv_s', rendered['inv_s'], prog_bar=True)
        elif hasattr(self.model.geometry, 'density'):
            self.log('train/beta', self.model.geometry.density.get_beta(), prog_bar=True)

        if self.precision == 16:
            self.log("precision/scale", self.trainer.precision_plugin.scaler.get_scale())

        return losses["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        img_size = self.datamodule.valset.image_shape
        self.model.timestep = -1

        # pass to pose corrector
        if hasattr(self.model, "pose_corrector"):
            batch["body_pose_before"] = batch["body_pose"].clone()
            batch["is_pose_corrected"] = True
            batch["body_pose"] = self.model.pose_corrector(batch["body_pose"], batch["idx"])

        rgb, depth, alpha, ray_normals, counter = self.model.render_image(batch, img_size)

        rgb_gt = batch["rgb"].reshape(-1, *img_size, 3)  # N, W, H, C
        rgb_with_bg_gt = batch["rgb_with_bg"].reshape(-1, *img_size, 3)  # N, W, H, C
        alpha_gt = batch["alpha"].reshape(-1, *img_size)
        loss_lpips, ssim = [0.] * 2

        # extra visualization for debugging
        if batch_idx == 0:
            rgb_ = rgb[..., [2, 1, 0]]
            self.save_image(f"animation/progression/{self.global_step:06d}.png", (rgb_[0].cpu().numpy() * 255).astype(np.uint8))

            ray_normals = (ray_normals.reshape(-1, *img_size, 3).clip(-1, 1) + 1) / 2
            self.save_image(f"animation/progression/{self.global_step:06d}_normals.png",
                            ((ray_normals * alpha_gt[...,None])[0].cpu().numpy() * 255).astype(np.uint8))
            self.save_grayscale_image(
                f"animation/progression/{self.global_step:06d}_depthmap.png",
                depth[0].cpu().numpy(),
                data_range=(0,6), cmap='jet',
            )
            rgb_with_bg_gt = rgb_with_bg_gt[..., [2, 1, 0]]
            rgb_with_bg_gt[alpha_gt > 0.5] = ray_normals[alpha_gt > 0.5]  # plot normal depth map onto image
            self.save_image(f"animation/progression/{self.global_step:06d}_normals_with_bg.png",
                            (rgb_with_bg_gt[0].cpu().numpy() * 255).astype(np.uint8))

            errmap_rgb = self.get_error_heatmap((rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy()[0] / (3**0.5)).to(self.rank)
            errmap_alpha = self.get_error_heatmap((alpha - alpha_gt).abs().cpu().numpy()[0]).to(self.rank)

            W, H = img_size
            self.save_image_grid(f"val/normaldepthmap/it{self.global_step}-{batch['idx'][0].item()}.png", [
                {'type': 'rgb', 'img': rgb_gt[..., [2, 1, 0]].view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': (ray_normals * alpha_gt[...,None]).view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': depth.view(W, H), 'kwargs': {'data_range': (0,6)}},
            ])
            self.save_image_grid(f"val/errmap/it{self.global_step}-{batch['idx'][0].item()}.png", [
                {'type': 'rgb', 'img': rgb_gt[..., [2, 1, 0]].view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': errmap_rgb.view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': errmap_alpha.view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
            ])

            # visualize novel pose
            batch["body_pose"][:] = 0
            batch["body_pose"][:, 2] = 0.5
            batch["body_pose"][:, 5] = -0.5

            dist = torch.sqrt(torch.square(batch["transl"]).sum(-1))
            batch["near"] = torch.ones_like(batch["rays_d"][..., 0]) * (dist - 1)
            batch["far"] = torch.ones_like(batch["rays_d"][..., 0]) * (dist + 1)

            rgb_cano, depth_cano, alpha_cano, ray_normals_cano, counter_cano = self.model.render_image(batch, img_size)
            ray_normals_cano = (ray_normals_cano.clip(-1, 1) + 1) / 2
            self.save_image_grid(f"val/cano_pose/it{self.global_step}-{batch['idx'][0].item()}_normaldepthmaps.png", [
                {'type': 'rgb', 'img': rgb_gt[..., [2, 1, 0]].view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': (ray_normals_cano * alpha_cano[...,None]).view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': depth_cano.view(W, H), 'kwargs': {'data_range': (0,6)}},
            ])
            self.save_image_grid(f"val/cano_pose/it{self.global_step}-{batch['idx'][0].item()}.png", [
                {'type': 'rgb', 'img': rgb_gt[..., [2, 1, 0]].view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rgb[..., [2, 1, 0]].view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rgb_cano[..., [2, 1, 0]].view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
            ])

            rgb_in = batch["rgb"].reshape(-1, *img_size, 3)
            loss_lpips = self.compute_lpips(rgb_in, rgb)
            ssim = self.compute_ssim(rgb_gt, rgb)

        return {
            "rgb_loss": (rgb - rgb_gt).square().mean(),
            "psnr": self.psnr(rgb, rgb_gt),
            "lpips": loss_lpips,
            "ssim": ssim,
            "counter_avg": counter.mean(),
            "counter_max": counter.max(),
            'idx': batch['idx'],
        }

    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['idx'].ndim == 1:
                    out_set[step_out['idx'].item()] = {
                        'psnr': step_out['psnr'], 'rgb_loss': step_out['rgb_loss'],
                        'lpips': step_out['lpips'], 'ssim': step_out['ssim'],
                    }
                # DDP
                else:
                    for oi, idx in enumerate(step_out['idx']):
                        out_set[idx[0].item()] = {
                            'psnr': step_out['psnr'][oi], 'rgb_loss': step_out['rgb_loss'][oi],
                            'lpips': step_out['lpips'][oi], 'ssim': step_out['ssim'][oi],
                        }
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            rgb_loss = torch.mean(torch.stack([o['rgb_loss'] for o in out_set.values()]))
            lpips = torch.mean(torch.stack([o['lpips'] for o in out_set.values()]))
            ssim = torch.mean(torch.stack([o['ssim'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)
            self.log('val/rgb_loss', rgb_loss, prog_bar=True, rank_zero_only=True)
            self.log('val/lpips', lpips, prog_bar=True, rank_zero_only=True)
            self.log('val/ssim', ssim, prog_bar=True, rank_zero_only=True)
            if self.global_step % self.config.export.export_every == 0:
                self.export('val', return_mesh=False)

    def export(self, split, return_mesh, idx=""):
        self.model.geometry.initialize(self.model.deformer.scene_aabb)
        mesh = self.model.export(self.config.export)
        try:
            self.save_mesh(
                f"{split}/{idx}-it{self.global_step}-{self.config.model.geometry.isosurface.resolution}.obj",
                **mesh,
                extract_large_component=True
            )
        except Exception as e:
            self.print("Error saving mesh: ", e)
        if return_mesh:
            return mesh

    def compute_lpips(self, gt, pred):
        img_size = self.datamodule.valset.image_shape
        size = 32
        stride = size
        patches_gt = gt.reshape(-1, *img_size, 3)
        patches_gt = patches_gt.permute(0, 3, 1, 2)  # N, C, H, W
        patches_gt = patches_gt.unfold(2, size, stride).unfold(3, size, stride)
        patches_gt = patches_gt.reshape(gt.shape[0], 3, -1, size, size).permute(0, 2, 3, 4, 1)  # N, P, HP, WP, C
        patches_out = pred
        patches_out = patches_out.permute(0, 3, 1, 2)  # N, C, H, W
        patches_out = patches_out.unfold(2, size, stride).unfold(3, size, stride)
        patches_out = patches_out.reshape(patches_out.shape[0], 3, -1, size, size).permute(0, 2, 3, 4, 1)  # N, P, HP, WP, C

        assert 0. - 1e-4 <= patches_out.min() and 1. + 1e-4 >= patches_out.max(), (patches_out.min(), patches_out.max())
        patches_out = scale_anything(patches_out, [0, 1], [-1, 1])
        assert 0. - 1e-4  <= patches_gt.min() and 1. + 1e-4 >= patches_gt.max(), (patches_gt.min(), patches_gt.max())
        patches_gt = scale_anything(patches_gt, [0, 1], [-1, 1])
        loss_lpips = self.loss_fn.lpips(
            patches_out[..., [2, 1, 0]].flatten(0, 1).permute(0, 3, 1, 2).clip(max=1),
            patches_gt[..., [2, 1, 0]].flatten(0, 1).permute(0, 3, 1, 2),
        ).mean()
        return loss_lpips

    def compute_ssim(self, gt, pred):
        ssim = self.ssim(
            scale_anything(gt.permute(0, 3, 1, 2), [0, 1], [-1, 1]),
            scale_anything(pred.permute(0, 3, 1, 2), [0, 1], [-1, 1]),
        )  # range is calculated from the data, as in Nerfacto
        return ssim

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        img_size = self.datamodule.testset.image_shape
        self.model.timestep = -1
        rgb, depth, alpha, ray_normals, counter = self.model.render_image(batch, img_size)

        # grab global_orient and transl from batch
        global_orient = batch['global_orient'].cpu().numpy()
        transl = batch['transl'].cpu().numpy()
        rot_c2w = batch['rot_c2w']
        global_orient = torch.from_numpy(R.from_rotvec(global_orient).as_matrix().squeeze()).float().to(self.rank)
        rot_normals = rot_c2w.squeeze().float().T @ global_orient
        rot_normals = torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]).to(self.rank).float() @ rot_normals
        rot_normals = torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).to(self.rank).float() @ rot_normals
        ray_normals = ray_normals.reshape(-1, 3)

        rgb_gt = batch["rgb"].reshape(-1, *img_size, 3)
        rgb_with_bg_gt = batch["rgb_with_bg"].reshape(-1, *img_size, 3)  # N, W, H, C
        alpha_gt = batch["alpha"].reshape(-1, *img_size)
        errmap = (rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy()[0] / (3 ** 0.5)
        errmap = self.get_error_heatmap(errmap).to(self.rank)
        W, H = img_size
        self.save_image_grid(f"test/errmap/it{self.global_step}-{batch['idx'][0].item()}.png", [
            {'type': 'rgb', 'img': rgb_gt[..., [2, 1, 0]].view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': rgb[..., [2, 1, 0]].view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': errmap.view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
        ])

        ray_normals = (ray_normals.reshape(-1, *img_size, 3).clip(-1, 1) + 1) / 2
        self.save_image_grid(f"test/normaldepthmap/it{self.global_step}-{batch['idx'][0].item()}.png", [
            {'type': 'rgb', 'img': rgb_gt[..., [2, 1, 0]].view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': (ray_normals * alpha_gt[...,None]).view(W, H, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': depth.view(W, H), 'kwargs': {'data_range': (0,6)}},
        ])

        self.save_grayscale_image(
            f"test/depthmap/it{self.global_step}-{batch['idx'][0].item()}.png",
            depth[0].cpu().numpy(),
            data_range=(0,6), cmap='jet',
        )

        rgb_with_bg = rgb[..., [2, 1, 0]].view(W, H, 3)
        self.save_image(f"test/rgb_with_bg/it{self.global_step}-{batch['idx'][0].item()}.png",
                        (rgb_with_bg.cpu().numpy() * 255).astype(np.uint8))

        normals_wo_bg = rgb[..., [2, 1, 0]].view(W, H, 3)
        normals_wo_bg[alpha_gt.squeeze() < 0.5] = 1.
        normals_wo_bg[alpha_gt.squeeze() >= 0.5] = ray_normals.squeeze()[alpha_gt.squeeze() >= 0.5]
        self.save_image(f"test/normals_wo_bg/it{self.global_step}-{batch['idx'][0].item()}.png",
                        (normals_wo_bg.cpu().numpy() * 255).astype(np.uint8))

        # rgb_with_bg_gt = rgb_with_bg_gt[..., [2, 1, 0]]
        # rgb_with_bg_gt[alpha_gt > 0.5] = ray_normals[alpha_gt > 0.5]  # plot normal depth map onto image
        # self.save_image(f"test/normals_with_bg/it{self.global_step}-{batch['idx'][0].item()}.png",
        #                 (rgb_with_bg_gt[0].cpu().numpy() * 255).astype(np.uint8))

        mesh_pred = self.export('test', return_mesh=True, idx=batch['idx'].item())

        out = {
            'idx': batch['idx'],
        }
        return out


    def test_epoch_end(self, out):
        pass