import pytorch_lightning as pl

from systems.utils import parse_optimizer, parse_scheduler, update_module_step
from utils.mixins import SaverMixin
from utils.misc import get_rank

from utils.misc import C


class BaseSystem(pl.LightningModule, SaverMixin):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.prepare(**kwargs)
 
    def prepare(self, **kwargs):
        pass

    def forward(self, batch):
        raise NotImplementedError

    def C(self, value, global_step=None, current_epoch=None):
        if global_step is None:
            global_step = self.global_step
        if current_epoch is None:
            current_epoch = self.current_epoch
        return C(value, global_step, current_epoch)

    def preprocess_data(self, batch, stage):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        self.preprocess_data(batch, 'train')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.val_dataloader().dataset
        self.preprocess_data(batch, 'validation')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.test_dataloader().dataset
        self.preprocess_data(batch, 'test')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset
        self.preprocess_data(batch, 'predict')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """

    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """

    def validation_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        ret = {
            'optimizer': optim,
        }
        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })
        return ret