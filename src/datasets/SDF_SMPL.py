import os
import pysdf
import torch
import trimesh
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import datasets
from SMPLX import SMPL
import ops.mesh as mesh_ops
from SMPL.serialization import load_model
from datasets.data_utils import load_smpl_param
from models.utils import get_predefined_smpl_rest_pose, get_bbox_from_smpl


def normalize_SMPLmodel(V, F, radius=1.0, normalization_mode="none"):
    V, F = mesh_ops.normalize(V, F, mode=normalization_mode)
    return V, F


class CustomDatasetBase():
    def setup(self, root, subject, split, config):
        self.config = config
        self.split = split
        self.rank = _get_rank()

        root = Path(root)
        self.root = root
        print('Pretraining SMPL-SDF with root: ', root)

        # # prepare image and mask
        # start = self.config.start
        # end = self.config.end + 1
        # skip = self.config.get("skip", 1)

        # ####### (START) PEOPLESNAPSHOT DATASET #######
        # subdir, suffix = "", ""
        # cached_path = root / Path(f"poses/anim_nerf_train.npz")
        # if cached_path and os.path.exists(cached_path):
        #     print(f"[{split}] Loading from", cached_path)
        #     self.smpl_params = load_smpl_param(cached_path)
        # else:
        #     print(f"[{split}] No optimized smpl found.")
        #     self.smpl_params = load_smpl_param(root / Path("poses.npz"))
        #     for k, v in self.smpl_params.items():
        #         if k != "betas":
        #             self.smpl_params[k] = v[start:end:skip]
        # self.smpl_params['scale'] = np.array([1.]).squeeze()
        # self.scale = self.smpl_params['scale']
        # self.shape = self.smpl_params['betas'].squeeze(0)
        # self.trans = self.smpl_params['transl']
        # self.poses = np.concatenate(
        #     [self.smpl_params['global_orient'], self.smpl_params['body_pose']],
        #     axis=1,
        # )
        # ####### (END) PEOPLESNAPSHOT DATASET #######

        ####### (START) XHUMANS DATASET #######
        subdir, suffix = "", ""
        cached_path = root / Path(f"smpl_camera_params.npz")
        self.smpl_params = dict(np.load(cached_path))
        self.smpl_params['scale'] = np.array([1.]).squeeze()
        self.scale = self.smpl_params['scale']
        self.shape = self.smpl_params['betas']
        self.trans = self.smpl_params['transl']
        self.poses = np.concatenate(
            [self.smpl_params['global_orient'], self.smpl_params['body_pose']],
            axis=1,
        )
        ####### (END) XHUMANS DATASET #######

        self.subdir = subdir

        # setup SMPL model
        self.mesh_path = os.path.join(root, f"A_pose{suffix}.obj")
        if os.path.exists(self.mesh_path):
            print('Loading mesh from: ', self.mesh_path)
            self.setup_SMPL()
        else:
            print('Getting vertices and faces from SMPL model')
            if isinstance(self.config.canonical_pose, str):
                body_pose_t = get_predefined_smpl_rest_pose(self.config.canonical_pose, device='cpu')
            else:
                body_pose_t = torch.zeros((1, 69), device='cpu')
                body_pose_t[:, 2]  = self.config.canonical_pose[0]
                body_pose_t[:, 5]  = self.config.canonical_pose[1]
                body_pose_t[:, 47] = self.config.canonical_pose[2]
                body_pose_t[:, 50] = self.config.canonical_pose[3]
            body_model = SMPL('../data/SMPLX/smpl', gender=self.config.gender)
            scale = torch.from_numpy(np.asarray(self.scale)).float().unsqueeze(0)
            betas = torch.from_numpy(self.shape).float().unsqueeze(0)
            smpl_outputs = body_model(
                scale=scale,
                betas=betas,
                body_pose=body_pose_t
            )
            self.V_norm = smpl_outputs.vertices.float()
            self.V_norm = self.V_norm.squeeze().cpu().detach().numpy()
            self.F_norm = torch.from_numpy(body_model.faces.astype(np.int64)).long()
            self.F_norm = self.F_norm.cpu().detach().numpy()
            mesh_ops.save(
                os.path.join(self.root / self.subdir, './normSMPL.ply'),
                V=self.V_norm,
                F=self.F_norm,
            )
            print(f'Stored normalized mesh in {os.path.join(self.root / self.subdir, "./normSMPL.ply")}')

        self.sdf_fn = pysdf.SDF(self.V_norm, self.F_norm)
        self.mesh_gt = trimesh.Trimesh(
            vertices=self.V_norm, faces=self.F_norm,
        )
        self.V_norm = torch.from_numpy(self.V_norm).float()
        self.F_norm = torch.from_numpy(self.F_norm).long()

        self.split = split
        self.near = self.config.get("near", None)
        self.far = self.config.get("far", None)

    def setup_SMPL(self,):
        V, F, pose = load_model(self.mesh_path)
        self.V_norm, self.F_norm = normalize_SMPLmodel(V, F, radius=-1, normalization_mode="none")
        # scale the mesh
        self.V_norm *= self.scale
        mesh_ops.save(
            os.path.join(self.root / self.subdir, './normSMPL.ply'),
            V=self.V_norm,
            F=self.F_norm,
        )

    def resample(self, num_samples, aabb):
        # compute aabb from self.V_norm on each axis
        aabb = torch.vstack([self.V_norm.min(0)[0] - 0.05, self.V_norm.max(0)[0] + 0.05]).cpu().numpy()

        sdfs = np.zeros((num_samples, 1))
        # surface
        points_surface = self.mesh_gt.sample(num_samples * 2 // 8)
        # perturb surface
        points_surface[:] += 0.05 * np.random.randn(num_samples * 2 // 8, 3)
        # random
        points_uniform = np.random.rand(num_samples * 6 // 8, 3)
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0])
        points_uniform = (points_uniform - 0.5) * scale + center
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[:] = -self.sdf_fn(points[:])[:,None].astype(np.float32)
        n = None

        if not os.path.exists(os.path.join(self.root / self.subdir, 'sample.ply')):
            print('Saving sample.ply')
            cloud=trimesh.PointCloud(points)
            cloud.export(os.path.join(self.root / self.subdir, 'sample.ply'))

        return points, sdfs, n


class SDF_SMPL_Dataset(Dataset, CustomDatasetBase):
    def __init__(self, root, subject, split, config):
        self.setup(root, subject, split, config)

    def get_SMPL_params(self):
        return {
            "betas": torch.from_numpy(self.shape.copy()).float(),
            "transl": torch.from_numpy(self.trans.copy()).float(),
            "global_orient": torch.from_numpy(self.poses[:, :3].copy()).float(),
            "body_pose": torch.from_numpy(self.poses[:, 3:].copy()).float(),
            "scale": torch.from_numpy(np.asarray(self.scale)).float(),
        }

    def __len__(self):
        return 100 if self.split == "train" else 1
        return self.poses.shape[0]

    def __getitem__(self, index):
        if len(self.poses.shape) == 1:
            return {
                "scale": torch.from_numpy(np.asarray(self.scale)).float(),
                "betas": torch.from_numpy(self.shape).float(),
                "transl": torch.from_numpy(self.trans).float(),
                "global_orient": torch.from_numpy(self.poses[:3]).float(),
                "body_pose": torch.from_numpy(self.poses[3:]).float(),
                'index': index
            }
        return {
            "scale": torch.from_numpy(np.asarray(self.scale)).float(),
            "betas": torch.from_numpy(self.shape).float(),
            "transl": torch.from_numpy(self.trans[index]).float(),
            "global_orient": torch.from_numpy(self.poses[index][:3]).float(),
            "body_pose": torch.from_numpy(self.poses[index][3:]).float(),
            'index': index
        }


@datasets.register('SMPL-SDF')
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        data_dir = os.path.abspath(config.dataroot)

        for split in ("train", "val", "test"):
            dataset = SDF_SMPL_Dataset(data_dir, config.subject, split, config.get(split))
            setattr(self, f"{split}set", dataset)
        self.config = config

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(self.trainset,
                              shuffle=True,
                              num_workers=self.config.train.num_workers,
                              persistent_workers=True and self.config.train.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            return DataLoader(self.valset,
                              shuffle=False,
                              num_workers=self.config.val.num_workers,
                              persistent_workers=True and self.config.val.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(self.testset,
                              shuffle=False,
                              num_workers=self.config.test.num_workers,
                              persistent_workers=True and self.config.test.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()