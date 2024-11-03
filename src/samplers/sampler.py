import cv2
import numpy as np

import samplers
from samplers.base import BaseSampler


@samplers.register('edge-sampler')
class EdgeSampler(BaseSampler):
    def setup(self, num_sample, ratio_mask=0.6, ratio_edge=0.3, kernel_size=32):
        assert ratio_mask >= 0.0
        assert ratio_edge >= 0.0
        assert ratio_edge + ratio_mask <= 1.0

        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

        self.num_mask = int(num_sample * ratio_mask)
        self.num_edge = int(num_sample * ratio_edge)
        self.num_rand = num_sample - self.num_mask - self.num_edge

    def sample(self, mask, *args):
        mask = mask.reshape(-1)

        # calculate the edge area
        mask_i = cv2.erode(mask, self.kernel)
        mask_o = cv2.dilate(mask, self.kernel)
        mask_e = mask_o - mask_i

        mask_loc, *_ = np.where(mask)
        edge_loc, *_ = np.where(mask_e)
        
        mask_idx = np.random.randint(0, len(mask_loc), self.num_mask)
        edge_idx = np.random.randint(0, len(edge_loc), self.num_edge)
        rand_idx = np.random.randint(0, len(mask), self.num_rand)

        mask_idx = mask_loc[mask_idx]
        edge_idx = edge_loc[edge_idx]

        indices = np.concatenate([mask_idx, edge_idx, rand_idx], axis=0)
        output = [mask[indices]]
        for d in args:
            d = d.reshape(len(mask), -1)
            output.append(d[indices])
        return output


@samplers.register('patch-sampler')
class PatchSampler(BaseSampler):
    def setup(self):
        self.n = self.config.num_patch
        self.patch_size = self.config.patch_size
        self.p = self.config.ratio_mask
        assert self.patch_size % 2 == 0, "patch size has to be even"

    def sample(self, mask, *args):
        patch = (self.patch_size, self.patch_size)
        shape = mask.shape[:2]
        if np.random.rand() < self.p:
            o = patch[0] // 2
            valid = mask[o:-o, o:-o] > 0
            (xs, ys) = np.where(valid)
            idx = np.random.choice(len(xs), size=self.n, replace=False)
            x, y = xs[idx], ys[idx]
        else:
            x = np.random.randint(0, shape[0] - patch[0], size=self.n)
            y = np.random.randint(0, shape[1] - patch[1], size=self.n)
        output = []
        for d in [mask, *args]:
            patches = []
            for xi, yi in zip(x, y):
                p = d[xi:xi + patch[0], yi:yi + patch[1]]
                patches.append(p)
            patches = np.stack(patches, axis=0)
            if patches.shape[-1] == 1: patches = patches.squeeze(-1)
            output.append(patches)
        return output