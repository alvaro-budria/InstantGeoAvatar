import sys
import math
import warnings
from bisect import bisect_right

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from pytorch_lightning.utilities.rank_zero import rank_zero_debug


class ChainedScheduler(lr_scheduler._LRScheduler):
    """Chains list of learning rate schedulers. It takes a list of chainable learning
    rate schedulers and performs consecutive step() functions belong to them by just
    one call.

    Args:
        schedulers (list): List of chained schedulers.

    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.09     if epoch == 0
        >>> # lr = 0.081    if epoch == 1
        >>> # lr = 0.729    if epoch == 2
        >>> # lr = 0.6561   if epoch == 3
        >>> # lr = 0.59049  if epoch >= 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = ChainedScheduler([scheduler1, scheduler2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, schedulers):
        for scheduler_idx in range(1, len(schedulers)):
            if (schedulers[scheduler_idx].optimizer != schedulers[0].optimizer):
                raise ValueError(
                    "ChainedScheduler expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                )
        self._schedulers = list(schedulers)
        self.optimizer = optimizer

    def step(self):
        for scheduler in self._schedulers:
            scheduler.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


class SequentialLR(lr_scheduler._LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given epoch.

    Args:
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.

    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if epoch == 0
        >>> # lr = 0.1     if epoch == 1
        >>> # lr = 0.9     if epoch == 2
        >>> # lr = 0.81    if epoch == 3
        >>> # lr = 0.729   if epoch == 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = SequentialLR(self.opt, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        for scheduler_idx in range(1, len(schedulers)):
            if (schedulers[scheduler_idx].optimizer != schedulers[0].optimizer):
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                )
        if (len(milestones) != len(schedulers) - 1):
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                "than the number of milestone points, but got number of schedulers {} and the "
                "number of milestones to be equal to {}".format(len(schedulers), len(milestones))
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
        else:
            self._schedulers[idx].step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


class ConstantLR(lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler decays the learning rate.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = ConstantLR(self.opt, factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1, verbose=False):
        if factor > 1.0 or factor < 0:
            raise ValueError('Constant multiplicative factor expected to be between 0 and 1.')

        self.factor = factor
        self.total_iters = total_iters
        super(ConstantLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters or
                (self.last_epoch != self.total_iters)):
            return [group['lr'] for group in self.optimizer.param_groups]

        if (self.last_epoch == self.total_iters):
            return [group['lr'] * (1.0 / self.factor) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
                for base_lr in self.base_lrs]


class LinearLR(lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters):
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]


custom_schedulers = ['ConstantLR', 'LinearLR']
def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    elif name in custom_schedulers:
        return getattr(sys.modules[__name__], name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split('.'):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        if name == 'geometry':
            for name, param in module.named_parameters():
                print(name)
            print('-'*20)
            ret = [param for name, param in module.named_parameters() if 'density' not in name]
            return ret
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model):
    if hasattr(config, 'params'):
        params = [
            {
                'params': get_parameters(model, name), 'name': name, **args
            } for name, args in config.params.items() if hasattr(model, name)
        ]
        params += [{'params': get_parameters(model.geometry, 'density'), 'name': 'density', **config.params.density}]
        # geometry encoding
        params += [{'params': get_parameters(model.geometry.encoding_with_network, 'encoding'), 'name': 'geometry_encoding', **config.params.geometry_encoding}]
        # geometry network
        params += [{'params': get_parameters(model.geometry.encoding_with_network, 'network'), 'name': 'geometry_network', **config.params.geometry_network}]
        rank_zero_debug('Specify optimizer params:', config.params)
    else:
        params = model.parameters()
    if config.name in ['FusedAdam']:
        import apex
        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler(config, optimizer):
    interval = config.get('interval', 'epoch')
    assert interval in ['epoch', 'step']
    if config.name == 'SequentialLR':
        scheduler = {
            'scheduler': SequentialLR(optimizer, [parse_scheduler(conf, optimizer)['scheduler'] for conf in config.schedulers], milestones=config.milestones),
            'interval': interval
        }
    elif config.name == 'Chained':
        scheduler = {
            'scheduler': ChainedScheduler([parse_scheduler(conf, optimizer)['scheduler'] for conf in config.schedulers]),
            'interval': interval
        }
    elif config.name == 'LambdaLR':
        scheduler = {
            'scheduler': get_scheduler(config.name)(optimizer, lambda x: eval(config.args.expression)),
            'interval': interval
        }
    else:
        scheduler = {
            'scheduler': get_scheduler(config.name)(optimizer, **config.args),
            'interval': interval
        }
    return scheduler


def update_module_step(m, epoch, global_step):
    if hasattr(m, 'update_step'):
        m.update_step(epoch, global_step)


def div_round_up(val, divisor):
    return int((val + divisor - 1) // divisor)


def next_multiple(val, divisor):
    return div_round_up(val, divisor) * divisor


def parse_tcnn_hash_grid_config(hash_encoding):
    encoding_config = hash_encoding.encoding_config
    b = encoding_config['per_level_scale']
    N0 = encoding_config['base_resolution']
    T = encoding_config['log2_hashmap_size']
    L = encoding_config['n_levels']
    F = encoding_config['n_features_per_level']
    codebook_size = 1 << T
    return b, N0, T, L, F, codebook_size


def get_start_indices_hash_grid(hash_encoding):
    """Given a tinycudann hash encoding, returns the indices that separate
    the levels in the 1D vector of weights."""
    b, N0, T, L, F, codebook_size = parse_tcnn_hash_grid_config(hash_encoding)

    resolution = dict()
    offset = 0
    lod_start_idxs, lod_sizes = [], []
    for l in range(L):
        grid_scale = 2 ** (l * math.log2(b)) * N0 - 1
        grid_resolution = math.ceil(grid_scale) + 1
        resolution[l+1] = grid_resolution
        params_in_level = int(grid_resolution ** 3)
        params_in_level = next_multiple(params_in_level, 8)
        Ml = min(params_in_level, codebook_size)
        lod_sizes.append(Ml)
        lod_start_idxs.append(offset)
        offset += Ml
    lod_start_idxs.append(offset)

    return lod_start_idxs, lod_sizes, F


def retrieve_hash_grid_weights(hash_encoding, retrieve_what: str = None):
    lod_start_idxs, lod_sizes, F = get_start_indices_hash_grid(hash_encoding)

    weights = list(hash_encoding.parameters())[0]
    if retrieve_what == 'gradients':
        weights = weights.grad.cpu().numpy().tolist()
    else:
        weights = weights.detach().cpu().numpy().tolist()

    lvl2weights = {
        i + 1: weights[lod_start_idxs[i] * F: lod_start_idxs[i+1] * F]
        for i in range(len(lod_start_idxs) - 1)
    }

    return lvl2weights
