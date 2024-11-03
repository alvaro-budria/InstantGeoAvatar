import torch
import torch.nn as nn

import models
from models.base import BaseModel


@models.register('smpl-embedding')
class SMPLEmbedding(BaseModel):
    """Allows optimizing SMPL parameters on the fly by gradient descent."""

    def setup(self):
        self.k2optimize = self.config.k2optimize

    def fill_parameters(self, smpl_params):
        # fill in init value
        for k, v in smpl_params.items():
            if k in self.k2optimize:
                v_ = v if len(v.shape) == 2 else v.unsqueeze(0)
                print('(fill SMPL embedding) k, shape: ', k, v_.shape)
                setattr(self, k, nn.Embedding.from_pretrained(v_, freeze=False))

    def forward(self, idx):
        out = {k: getattr(self, k)(idx) for k in self.k2optimize if k != 'betas'}
        return {
            **out,
            'betas': getattr(self, 'betas')(idx - idx),
        }

    def prepare_batch(self, batch, substitute):
        idx = batch["idx"] + batch.get("idx_start", 0)
        body_params = self(idx)
        for k in self.k2optimize:
            assert batch[k].shape == body_params[k].shape
            batch[k] = (1-int(substitute)) * batch[k] + int(substitute) * body_params[k]

        # update near & far with refined SMPL
        dist = torch.norm(batch["transl"], dim=-1, keepdim=True).detach()
        batch["near"][:] = dist - 1
        batch["far"][:] = dist + 1
        return batch

    def tv_loss(self, idx):
        loss = 0.
        N = len(self.global_orient.weight)
        idx_p = (idx - 1).clip(min=0)
        idx_n = (idx + 1).clip(max=N - 1)
        for k in ['body_pose', 'global_orient', 'transl']:
            loss = loss + (getattr(self, k)(idx) - getattr(self, k)(idx_p)).square().mean()
            loss = loss + (getattr(self, k)(idx_n) - getattr(self, k)(idx)).square().mean()
        return loss