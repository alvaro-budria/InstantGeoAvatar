import os
import cv2
import csv
import glob
import torch
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import datasets
import samplers
from datasets.data_utils import make_rays


class XHumansDatasetBase():
    def setup(self, root, subject, split, config):
        self.config = config
        self.split = split
        self.rank = _get_rank()

        root = Path(root)
        self.root = root

        # Load csv file with index limits
        csv_file = root / Path(f"{subject}_index_limits.csv")
        myFile = open(csv_file, 'r')
        reader = csv.DictReader(myFile)
        myList = list()
        for dictionary in reader:
            myList.append(dictionary)
        if split in ["train", "val"]:
            myList = [x for x in myList if x['type'] == 'train']
        elif split == "test":
            myList = [x for x in myList if x['type'] == 'test']
        else:
            raise ValueError(f"split {split} not recognized")
        start = min([int(x['start']) for x in myList])
        end = max([int(x['end']) for x in myList]) + 1 if split != "val" else start + 1
        skip = self.config.get("skip", 1)
        self.frame_indices = np.arange(start, end, skip)

        smpl_camera_params = np.load(str(root / Path("smpl_camera_params.npz")))
        Ks = smpl_camera_params["intrinsic"]
        c2ws = np.linalg.inv(smpl_camera_params["extrinsic"])  # smpl_camera_params["extrinsic"] contains w2c
        height = smpl_camera_params["height"]
        width = smpl_camera_params["width"]

        self.downscale = self.config.downscale
        if self.downscale > 1:
            height = int(height / self.downscale)
            width = int(width / self.downscale)
            Ks[:, :2] /= self.downscale

        # select camera parameters
        self.Ks = Ks[start:end:skip]
        self.c2ws = c2ws[start:end:skip]

        # prepare image and mask
        self.img_lists = sorted(glob.glob(f"{root}/images/*.png"))[start:end:skip]
        self.msk_lists = sorted(glob.glob(f"{root}/masks/*.png"))[start:end:skip]
        self.timesteps = np.arange(len(self.img_lists))

        self.smpl_params = {k: v for (k, v) in smpl_camera_params.items()
                            if k not in ["intrinsic", "extrinsic", "height", "width", "gender"]}
        for k, v in self.smpl_params.items():
            if k != "betas":
                self.smpl_params[k] = v[start:end:skip]
        self.near = self.config.get("near", None)
        self.far = self.config.get("far", None)
        self.image_shape = (height, width)
        if split == "train":
            self.sampler = samplers.make(self.config.sampler.name, self.config.sampler)
            self.patch_size = self.sampler.patch_size
            self.num_patches = self.sampler.n

class XHumansDataset(Dataset, XHumansDatasetBase):
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
        msk = cv2.imread(self.msk_lists[idx], cv2.IMREAD_GRAYSCALE)
        if self.downscale > 1:
            img = cv2.resize(img, dsize=None, fx=1/self.downscale, fy=1/self.downscale)
            msk = cv2.resize(msk, dsize=None, fx=1/self.downscale, fy=1/self.downscale)

        img = (img[..., :3] / 255).astype(np.float32)
        msk = (msk / 255).astype(np.float32)
        msk = (msk > 0.5).astype(np.float32)
        # apply mask
        if self.split == "train":
            bg_color = np.random.rand(*img.shape).astype(np.float32)
            img = img * msk[..., None] + (1 - msk[..., None]) * bg_color
        else:
            bg_color = np.ones_like(img).astype(np.float32)
            img_with_bg = img.copy()
            img = img * msk[..., None] + (1 - msk[..., None])

        K = self.Ks[idx]
        c2w = self.c2ws[idx]
        height, width = self.image_shape
        self.rays_o, self.rays_d = make_rays(K, c2w, height, width)
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
            "betas": self.smpl_params["betas"],
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
            datum.update({'rot_c2w': c2w[:3, :3]})
        else:
            datum.update({
                "patch_size": self.patch_size,
                "num_patches": self.num_patches,
            })
        if self.near is not None and self.far is not None:
            datum["near"] = np.ones_like(rays_d[..., 0]) * self.near
            datum["far"] = np.ones_like(rays_d[..., 0]) * self.far
        else:
            # distance from camera (0, 0, 0) to midhip
            dist = np.sqrt(np.square(self.smpl_params["transl"][idx]).sum(-1))
            datum["near"] = np.ones_like(rays_d[..., 0]) * (dist - 1)
            datum["far"] = np.ones_like(rays_d[..., 0]) * (dist + 1)
        return datum


@datasets.register('x-humans')
class XHumansDataModule(pl.LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        data_dir = os.path.abspath(config.dataroot)

        for split in ("train", "val", "test"):
            dataset = XHumansDataset(data_dir, config.subject, split, config.get(split))
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