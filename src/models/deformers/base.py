import torch.nn as nn

from utils.misc import get_rank


class BaseDeformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.setup()

    def setup(self):
        raise NotImplementedError

    def update_step(self, epoch, global_step):
        pass

    def train(self, mode=True):
        return super().train(mode=mode)

    def eval(self):
        return super().eval()