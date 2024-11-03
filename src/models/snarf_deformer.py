import os
import torch

from pytorch_lightning.utilities.rank_zero import rank_zero_info

import models
from SMPLX import SMPL
from utils.misc import get_rank
from models.deformers.fast_SNARF_torch import FastSNARF_forward
from models.utils import get_predefined_smpl_rest_pose, get_bbox_from_smpl
from models.network_utils import get_embedder_Hann


@models.register('fast-snarf')
class SNARFDeformer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.setup()

    def setup(self):
        smpl_model_path = os.path.abspath(self.config.model_path)
        print(f"Loading SMPL model from {smpl_model_path}")
        self.body_model = SMPL(smpl_model_path, gender=self.config.gender)
        self.deformer = FastSNARF_forward(self.config)
        self.initialized = False
        self.current_epoch = 0

    def initialize(self, betas, device):
        if isinstance(self.config.canonical_pose, str):
            body_pose_t = get_predefined_smpl_rest_pose(self.config.canonical_pose, device=device)
        else:
            body_pose_t = torch.zeros((1, 69), device=device)
            body_pose_t[:, 2] = self.config.canonical_pose[0]
            body_pose_t[:, 5] = self.config.canonical_pose[1]
            body_pose_t[:, 47] = self.config.canonical_pose[2]
            body_pose_t[:, 50] = self.config.canonical_pose[3]

        smpl_outputs = self.body_model(betas=betas[:1], body_pose=body_pose_t)
        self.transforms_inv_t = torch.inverse(smpl_outputs.A.float().detach())
        self.vs_template = smpl_outputs.vertices

        # initialize SNARF
        self.deformer.device = device
        self.deformer.switch_to_grid(
            resolution=self.config.resolution,
            smpl_verts=smpl_outputs.vertices.float().detach(),
            smpl_weights=self.body_model.lbs_weights.clone()[None].detach(),
            use_smpl=True,
        )
        self.scene_aabb = get_bbox_from_smpl(smpl_outputs.vertices.detach())

        self.dtype = torch.float32
        self.deformer.lbs_voxel_final = self.deformer.lbs_voxel_final.type(self.dtype)  # 24, 16, 64, 64
        self.deformer.grid_denorm = self.deformer.grid_denorm.type(self.dtype)
        self.deformer.scale = self.deformer.scale.type(self.dtype)
        self.deformer.offset = self.deformer.offset.type(self.dtype)
        self.deformer.scale_kernel = self.deformer.scale_kernel.type(self.dtype)
        self.deformer.offset_kernel = self.deformer.offset_kernel.type(self.dtype)

        # define collection of locations at the vertices of the voxel grid
        shape = self.deformer.lbs_voxel_final.shape[2:]
        self.deformer.voxel_locations = torch.stack(torch.meshgrid([torch.arange(s, device=device) for s in shape]), dim=-1).float()
        # normalize the locations to [0, 1] along each axis
        self.deformer.voxel_locations[..., 0] /= shape[0] - 1
        self.deformer.voxel_locations[..., 1] /= shape[1] - 1
        self.deformer.voxel_locations[..., 2] /= shape[2] - 1
        self.deformer.voxel_locations = self.deformer.voxel_locations.type(self.dtype)

    def prepare_deformer(self, smpl_params, delta_weights_predictor=None):
        device = smpl_params["betas"].device
        self.body_model = self.body_model.to(device)

        if not self.initialized:
            self.initialize(smpl_params["betas"], smpl_params["betas"].device)
            self.initialized = True

        smpl_outputs = self.body_model(
            betas=smpl_params["betas"],
            body_pose=smpl_params["body_pose"],
            global_orient=smpl_params["global_orient"],
            transl=smpl_params["transl"],
        )
        s2w = smpl_outputs.A[:, 0].float()
        w2s = torch.inverse(s2w)

        transforms = (w2s[:, None] @ smpl_outputs.A.float() @ self.transforms_inv_t).type(self.dtype)  # (B, 24, 4, 4)

        self.deformer.precompute(transforms)

        self.w2s = w2s
        self.vertices = (smpl_outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        self.transforms = transforms
        self.smpl_outputs = smpl_outputs
        self.smpl_params = smpl_params

    def transform_rays_w2s(self, rays_o, rays_d):
        """transform rays from world to smpl coordinate system"""
        w2s = self.w2s

        rays_o = (rays_o @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        rays_d = (rays_d @ w2s[:, :3, :3].permute(0, 2, 1)).to(rays_d)
        d = torch.norm(rays_o, dim=-1)
        near = d - 1
        far = d + 1
        return rays_o, rays_d, near, far

    def get_bbox_deformed(self):
        voxel = self.deformer.voxel_d[0].reshape(3, -1)
        return [voxel.min(dim=1).values, voxel.max(dim=1).values]

    def deform(self, points, eval_mode):
        """Warp points from normalized space to canonical space"""
        point_size = points.shape[0]
        betas = self.smpl_outputs.betas
        batch_size = betas.shape[0]
        points = points.reshape(batch_size, -1, 3)
        deformer_out = self.deformer(points, cond=None, bone_transf=self.transforms, eval_mode=eval_mode)
        pts_canonical = deformer_out["result"].reshape(point_size, -1, 3)
        valid = deformer_out["valid_ids"].reshape(point_size, -1)

        return pts_canonical, valid

    def __call__(self, points, eval_mode):
        with torch.set_grad_enabled(not eval_mode):
            return self.deform(points.type(self.dtype), eval_mode)