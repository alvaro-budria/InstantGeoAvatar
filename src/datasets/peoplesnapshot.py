
import os
import cv2
import glob
import torch
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import samplers
import datasets
from datasets.data_utils import make_rays, load_smpl_param


class PeopleSnapshotDatasetBase():
    def setup(self, root, subject, split, config):
        self.config = config
        self.split = split
        self.rank = _get_rank()

        root = Path(root)
        self.root = root
        camera = np.load(str(root / Path("cameras.npz")))
        K = camera["intrinsic"]
        c2w = np.linalg.inv(camera["extrinsic"])
        height = camera["height"]
        width = camera["width"]

        self.downscale = self.config.downscale
        if self.downscale > 1:
            height = int(height / self.downscale)
            width = int(width / self.downscale)
            K[:2] /= self.downscale

        # prepare image and mask
        start = self.config.start
        end = self.config.end + 1
        skip = self.config.get("skip", 1)
        self.img_lists = sorted(glob.glob(f"{root}/images/*.png"))[start:end:skip]
        self.msk_lists = sorted(glob.glob(f"{root}/masks/*.npy"))[start:end:skip]
        self.frame_indices = np.arange(start, end, skip)
        print('frame_indices', self.frame_indices, len(self.frame_indices))
        self.timesteps = np.arange(len(self.img_lists))

        refine = self.config.get("refine", False)
        if refine: # fix model and optimize SMPL
            cached_path = root / "poses/anim_nerf_test.npz"
        else:
            if os.path.exists(root / Path(f"poses/anim_nerf_{split}.npz")):
                cached_path = root / Path(f"poses/anim_nerf_{split}.npz")
            elif os.path.exists(root / Path(f"poses/{split}.npz")):
                cached_path = root / Path(f"poses/{split}.npz")
            else:
                cached_path = None

        if cached_path and os.path.exists(cached_path):
            print(f"[{split}] Loading from", cached_path)
            self.smpl_params = load_smpl_param(cached_path)
        else:
            print(f"[{split}] No optimized smpl found.")
            self.smpl_params = load_smpl_param(root / Path("poses.npz"))
            for k, v in self.smpl_params.items():
                if k != "betas":
                    self.smpl_params[k] = v[start:end:skip]
                    assert len(self.smpl_params[k]) == len(self.img_lists), f"{k} {len(self.smpl_params[k])} != {len(self.img_lists)}"

        self.split = split
        self.downscale = self.config.downscale
        self.near = self.config.get("near", None)
        self.far = self.config.get("far", None)
        self.image_shape = (height, width)
        if split == "train":
            self.sampler = samplers.make(self.config.sampler.name, self.config.sampler)
            self.patch_size = self.sampler.patch_size

        self.rays_o, self.rays_d = make_rays(K, c2w, height, width)

class PeopleSnapshotDataset(Dataset, PeopleSnapshotDatasetBase):
    def __init__(self, root, subject, split, config):
        self.setup(root, subject, split, config)

    def get_SMPL_params(self):
        return {
            k: torch.from_numpy(v.copy()) for k, v in self.smpl_params.items()
        }

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_lists[idx])
        msk = np.load(self.msk_lists[idx])
        if self.downscale > 1:
            img = cv2.resize(img, dsize=None, fx=1/self.downscale, fy=1/self.downscale)
            msk = cv2.resize(msk, dsize=None, fx=1/self.downscale, fy=1/self.downscale)

        img = (img[..., :3] / 255).astype(np.float32)
        msk = msk.astype(np.float32)
        # apply mask
        if self.split == "train":
            bg_color = np.random.rand(*img.shape).astype(np.float32)
            img = img * msk[..., None] + (1 - msk[..., None]) * bg_color
        else:
            bg_color = np.ones_like(img).astype(np.float32)
            img_with_bg = img.copy()
            img = img * msk[..., None] + (1 - msk[..., None])

        if self.split == "train":
            (msk, img, rays_o, rays_d, bg_color) = \
                    self.sampler.sample(msk, img, self.rays_o, self.rays_d, bg_color)
        else:
            rays_o = self.rays_o.reshape(-1, 3)
            rays_d = self.rays_d.reshape(-1, 3)
            img_with_bg = img_with_bg.reshape(-1, 3)
            img = img.reshape(-1, 3)
            msk = msk.reshape(-1)

        datum = {
            # NeRF
            "rgb": img.astype(np.float32),
            "rays_o": rays_o,
            "rays_d": rays_d,

            # SMPL parameters
            "betas": self.smpl_params["betas"][0],
            "global_orient": self.smpl_params["global_orient"][idx],
            "body_pose": self.smpl_params["body_pose"][idx],
            "transl": self.smpl_params["transl"][idx],

            # auxiliary
            "alpha": msk,
            "bg_color": bg_color,
            "idx": idx,
            "timestep": self.timesteps[idx] if self.split == "train" else -1,
            "frame_idx": self.frame_indices[idx],
        }
        if self.split != "train":
            datum.update({'rgb_with_bg': img_with_bg.astype(np.float32)})
            datum.update({'rot_c2w': np.eye(3)})
        if self.near is not None and self.far is not None:
            datum["near"] = np.ones_like(rays_d[..., 0]) * self.near
            datum["far"] = np.ones_like(rays_d[..., 0]) * self.far
        else:
            # distance from camera (0, 0, 0) to midhip
            dist = np.sqrt(np.square(self.smpl_params["transl"][idx]).sum(-1))
            datum["near"] = np.ones_like(rays_d[..., 0]) * (dist - 1)
            datum["far"] = np.ones_like(rays_d[..., 0]) * (dist + 1)
        return datum


@datasets.register('peoplesnapshot')
class PeopleSnapshotDataModule(pl.LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        data_dir = os.path.abspath(config.dataroot)

        for split in ("train", "val", "test"):
            dataset = PeopleSnapshotDataset(data_dir, config.subject, split, config.get(split))
            setattr(self, f"{split}set", dataset)
        self.config = config

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(
                self.trainset,
                shuffle=True,
                num_workers=self.config.train.num_workers,
                persistent_workers=True and self.config.train.num_workers > 0,
                pin_memory=True,
                batch_size=1,
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            return DataLoader(
                self.valset,
                shuffle=False,
                num_workers=self.config.val.num_workers,
                persistent_workers=True and self.config.val.num_workers > 0,
                pin_memory=True,
                batch_size=1,
            )
        else:
            return super().test_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(
                self.testset,
                shuffle=False,
                num_workers=self.config.test.num_workers,
                persistent_workers=True and self.config.test.num_workers > 0,
                pin_memory=True,
                batch_size=1,
            )
        else:
            return super().test_dataloader()
