import torch
import torch.nn.functional as F

from lpips import LPIPS

import losses
from losses.base import BaseLoss
from utils.misc import C


@losses.register('psnr')
class PSNR(BaseLoss):
    def setup(self) -> None:
        pass

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        value = (inputs - targets)**2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return -10 * torch.log10(torch.mean(value))
        elif reduction == 'none':
            return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


@losses.register('instantgeoavatar-loss')
class InstantAvatar(BaseLoss):
    def setup(self) -> None:
        self.lpips = LPIPS(net="vgg", pretrained=True)
        for param in self.lpips.parameters():
            param.requires_grad=False

    def forward(self, predicts, targets, global_step, current_epoch):
        losses = {}
        loss = 0

        loss_rgb = F.mse_loss(predicts["comp_rgb"].squeeze(), targets["rgb"].squeeze(), reduction="mean")
        loss += C(self.config.lambda_rgb) * loss_rgb
        losses["mse_loss"] = loss_rgb

        loss_alpha = F.mse_loss(predicts["opacity"].squeeze(), targets["alpha"].squeeze())
        loss += C(self.config.lambda_mask) * loss_alpha
        losses["loss_opacity"] = loss_alpha

        loss_eikonal = ((torch.linalg.norm(predicts['sdfgrad'], ord=2, dim=-1) - 1.)**2).mean()
        loss += C(self.config.lambda_eikonal) * loss_eikonal
        losses["loss_eikonal"] = loss_eikonal

        if 'normals' in predicts:
            normal_starts, normal_mids, normal_ends = predicts['normals']
            if 'valid_samples' in predicts:
                w = predicts['valid_samples'][:-1] * predicts['valid_samples'][1:]
            elif 'w' in predicts:
                w = predicts['w']
            else:
                raise ValueError('No valid weighting scheme found for flat surface regularization loss.')
            loss_flat_surface = + (w[...,None] * (normal_mids - normal_starts)**2).mean() \
                                + (w[...,None] * (normal_mids - normal_ends  )**2).mean()
            loss += C(self.config.lambda_flat_surface) * loss_flat_surface
            losses["loss_flat_surface"] = loss_flat_surface

        if self.config.get("lambda_lpips", 0) > 0 and len(predicts["comp_rgb"].shape) == 5:
            loss_lpips = self.lpips(predicts["comp_rgb"][..., [2, 1, 0]].flatten(0, 1).permute(0, 3, 1, 2).clip(max=1),
                                    targets["rgb"][..., [2, 1, 0]].flatten(0, 1).permute(0, 3, 1, 2)).sum()
            losses["loss_lpips"] = loss_lpips
            loss += loss_lpips * C(self.config.lambda_lpips)
        loss_lpips = self.lpips(predicts["comp_rgb"][..., [2, 1, 0]].permute(0, 3, 1, 2).clip(max=1),
                                targets["rgb"].squeeze(0)[..., [2, 1, 0]].permute(0, 3, 1, 2)).sum()
        losses["loss_lpips"] = loss_lpips
        loss += loss_lpips * C(self.config.lambda_lpips)

        if C(self.config.get("lambda_depth_reg", 0)) > 0 and len(predicts["comp_rgb"].shape) == 5:
            alpha_sum = predicts["opacity"].sum(dim=(-1, -2))
            depth_avg = (predicts["depth"] * predicts["opacity"]).sum(dim=(-1, -2)) / (alpha_sum + 1e-3)
            loss_depth_reg = predicts["opacity"] * (predicts["depth"] - depth_avg[..., None, None]).abs()
            loss_depth_reg = loss_depth_reg.mean()
            losses["loss_depth_reg"] = loss_depth_reg
            loss += C(self.config.lambda_depth_reg) * loss_depth_reg

        OFFSET = 0.313262
        reg_alpha   = (-torch.log(torch.exp(-predicts["opacity"]) + torch.exp(predicts["opacity"] - 1))).mean() + OFFSET
        reg_density = (-torch.log(torch.exp(-predicts["weights"]) + torch.exp(predicts["weights"] - 1))).mean() + OFFSET
        losses["reg_alpha"] = reg_alpha
        losses["reg_density"] = reg_density
        loss += self.config.lambda_reg * reg_alpha
        loss += self.config.lambda_reg * reg_density

        losses["loss"] = loss
        return losses