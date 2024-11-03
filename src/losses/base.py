import torch.nn as nn

import losses
from utils.misc import get_rank


@losses.register('base-loss')
class BaseLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.setup()

    def setup(self):
        print('WARNING (BaseLoss): Not implemented yet!')
        pass

    def update_step(self, epoch, global_step):
        pass