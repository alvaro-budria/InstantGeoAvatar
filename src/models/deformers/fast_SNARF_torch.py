import os
import torch
from torch import einsum
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from models.deformers.base import BaseDeformer
from models.deformers.utils import bmv, skinning_mask, query_weights_smpl

cuda_dir = os.path.join(os.path.dirname(__file__), "cuda")
fuse_kernel = load(name='fuse_cuda',
                   extra_cuda_cflags=[],
                   sources=[f'{cuda_dir}/fuse_kernel/fuse_cuda.cpp',
                            f'{cuda_dir}/fuse_kernel/fuse_cuda_kernel_fast.cu'])
filter_cuda = load(name='filter',
                   sources=[f'{cuda_dir}/filter/filter.cpp',
                            f'{cuda_dir}/filter/filter.cu'])
precompute_cuda = load(name='precompute',
                       sources=[f'{cuda_dir}/precompute/precompute.cpp',
                                f'{cuda_dir}/precompute/precompute.cu'])


class FastSNARF_forward(BaseDeformer):
    def setup(self):
        self.soft_blend = 20

        self.init_bones = [0, 1, 2, 4, 5, 10, 11, 12, 15, 16, 17, 18, 19]
        self.init_bones_cuda = torch.tensor(self.init_bones).cuda().int()

        # the bounding box should be slightly larger than the actual mesh
        self.global_scale = 1.2
        self.version = self.config.get("version", 1)

    def forward(self, x_deformed, cond, bone_transf, eval_mode):
        """Given a deformed point return its canonical correspondence
        Args:
            x_deformed (tensor): deformed points in batch. Shape: [B, N, D]
            cond (dict): conditional input.
            bone_transf (tensor): bone transformation matrices. Shape: [B, J, D+1, D+1]
        Returns:
            search_out (dict): contains canonical correspondences (tensor: [B, N, I, D]) and other useful outputs.
        """
        search_out = self.search(x_deformed, bone_transf)
        if eval_mode:
            return search_out

        x_canon_opt = search_out['result']
        # both versions should deliver the same result in the same amount of time
        # https://github.com/tijiang13/InstantAvatar/issues/26
        if self.version == 1:
            x_canon_opt = x_canon_opt.detach()
            x_canon_opt[~search_out['valid_ids']] = 0
            n_batch, n_point, n_init, n_dim = x_canon_opt.shape

            mask = search_out['valid_ids']
            # x_canon_opt.requires_grad = True
            x_deformed_opt = self.forward_skinning(x_canon_opt, cond, bone_transf, mask=mask)

            grad_inv = search_out['J_inv'][search_out['valid_ids']]
            correction = x_deformed_opt - x_deformed_opt.detach()
            correction = bmv(-grad_inv, correction.unsqueeze(-1)).squeeze(-1)

            # trick for implicit diff with autodiff:
            # x_canon = x_canon_opt + 0 and x_canon' = correction'
            x_canon = x_canon_opt
            x_canon[search_out['valid_ids']] += correction
            x_canon = x_canon.reshape(n_batch, n_point, n_init, n_dim)
        else:
            mask = search_out['valid_ids']
            weights = self.query_weights(x_canon_opt, cond, mask=mask)
            T = einsum("pn,nij->pij", weights[mask], bone_transf[0])
            pts = x_deformed[..., None, :].expand(1, -1, len(self.init_bones), 3)[mask]
            x_canon = torch.zeros_like(x_canon_opt)
            x_canon[mask] = ((pts - T[:, :3, 3]).unsqueeze(-2) @ T[:, :3, :3]).squeeze(1)

        search_out['result'] = x_canon
        return search_out

    def precompute(self, bone_transf):
        b, c, d, h, w = bone_transf.shape[0], 3, self.resolution // 4, self.resolution, self.resolution
        voxel_d = torch.zeros((b, 3, d, h, w), device=bone_transf.device)
        voxel_J = torch.zeros((b, 12, d, h, w), device=bone_transf.device)
        precompute_cuda.precompute(self.lbs_voxel_final, bone_transf, voxel_d, voxel_J, self.offset_kernel, self.scale_kernel)
        self.voxel_d = voxel_d
        self.voxel_J = voxel_J

    def search(self, x_deformed, bone_transf):
        """Search correspondences through iterative root finding.
        Args:
            x_deformed (tensor): deformed points in batch. shape: [B, N, D]
            bone_transf (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            result (dict): the canonical correspondences (tensor: [B, N, I, D])
                           and identifiers of converged points (tensor: [B, N, I])
        """
        with torch.no_grad():
            result = self.broyden_cuda(x_deformed, self.voxel_d, self.voxel_J, bone_transf)
        return result

    def broyden_cuda(self, x_deformed_target, voxel, voxel_J_inv, bone_transf, cvg_thresh=1e-5, dvg_thresh=1e-1):
        b, n, _ = x_deformed_target.shape
        n_init = self.init_bones_cuda.shape[0]

        x_canonical_init_IN = torch.zeros((b, n, n_init, 3), device=x_deformed_target.device, dtype=torch.float32)
        J_inv_init_IN = torch.zeros((b, n, n_init, 3, 3), device=x_deformed_target.device, dtype=torch.float32)
        is_valid = torch.zeros((b, n, n_init), device=x_deformed_target.device, dtype=torch.bool)
        fuse_kernel.fuse_broyden(x_canonical_init_IN, x_deformed_target, voxel, voxel_J_inv, bone_transf,
                                 self.init_bones_cuda, True, J_inv_init_IN,
                                 is_valid, self.offset_kernel,
                                 self.scale_kernel, cvg_thresh, dvg_thresh)
        mask = filter_cuda.filter(x_canonical_init_IN, is_valid)
        return {
            "result": x_canonical_init_IN,
            'valid_ids': mask,
            'J_inv': J_inv_init_IN
        }

    def forward_skinning(self, x_canon, cond, bone_transf, mask=None):
        """Canonical point -> deformed point
        Args:
            x_canon (tensor): canonical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            bone_transf (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            x_deformed (tensor): deformed point. shape: [B, N, D]
        """
        weights = self.query_weights(x_canon, cond, mask=mask)
        return skinning_mask(x_canon[mask], weights[mask], bone_transf)

    def switch_to_grid(self, resolution=32, smpl_verts=None, smpl_weights=None, use_smpl=False):
        self.resolution = resolution
        # convert to voxel grid
        device = self.device
        b, c, d, h, w = 1, 24, resolution // 4, resolution, resolution
        self.ratio = h / d
        x_range = (torch.linspace(-1, 1, steps=w, device=device)).view(1, 1, 1, w).expand(1, d, h, w)
        y_range = (torch.linspace(-1, 1, steps=h, device=device)).view(1, 1, h, 1).expand(1, d, h, w)
        z_range = (torch.linspace(-1, 1, steps=d, device=device)).view(1, d, 1, 1).expand(1, d, h, w)
        grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(b, 3, -1).permute(0, 2, 1)

        gt_bbox = torch.cat([smpl_verts.min(dim=1).values, smpl_verts.max(dim=1).values], dim=0).to(device)
        offset = (gt_bbox[0] + gt_bbox[1])[None, None, :] * 0.5
        scale = (gt_bbox[1] - gt_bbox[0]).max() / 2 * self.global_scale

        corner = torch.ones_like(offset[0]) * scale
        corner[0, 2] /= self.ratio
        min_vert = (offset - corner).reshape(1, 3)
        max_vert = (offset + corner).reshape(1, 3)
        self.scene_aabb = torch.cat([min_vert, max_vert], dim=0)

        self.register_buffer('scale', scale)
        self.register_buffer('offset', offset)

        self.register_buffer('offset_kernel', -self.offset)
        scale_kernel = torch.zeros_like(self.offset)
        scale_kernel[...] = 1. / self.scale
        scale_kernel[:, :, -1] = scale_kernel[:, :, -1] * self.ratio
        self.register_buffer('scale_kernel', scale_kernel)

        def normalize(x):
            x_normalized = x.clone()
            x_normalized -= self.offset
            x_normalized /= self.scale
            x_normalized[..., -1] *= self.ratio
            return x_normalized

        def denormalize(x):
            x_denormalized = x.clone()
            x_denormalized[..., -1] /= self.ratio
            x_denormalized *= self.scale
            x_denormalized += self.offset
            return x_denormalized

        self.normalize = normalize
        self.denormalize = denormalize

        grid_denorm = self.denormalize(grid)

        if use_smpl:
            weights = query_weights_smpl(
                grid_denorm,
                smpl_verts=smpl_verts.detach().clone(),
                smpl_weights=smpl_weights.detach().clone(),
                resolution=resolution,
            ).detach().clone()
        else:
            weights = self.query_weights(grid_denorm, {}, None)

        self.register_buffer('lbs_voxel_final', weights.detach())
        self.register_buffer('grid_denorm', grid_denorm)

        def query_weights(x_canon, cond=None, mask=None, mode='bilinear'):
            shape = x_canon.shape
            N = 1
            x_canon = x_canon.view(1, -1, 3)
            w = F.grid_sample(
                              self.lbs_voxel_final.expand(N, -1, -1, -1, -1),
                              self.normalize(x_canon)[:, :, None, None],
                              align_corners=True,
                              mode=mode,
                              padding_mode='border')
            w = w.squeeze(-1).squeeze(-1).permute(0, 2, 1)
            w = w.view(*shape[:-1], -1)
            return w
        self.query_weights = query_weights