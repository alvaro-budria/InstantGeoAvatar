import os
import sys
import glob
import torch
import logging
import argparse
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

import systems
import datasets
from utils.misc import load_config
from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
    parser.add_argument(
        '--resume_weights_only',
        action='store_true',
        help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pretrain_SMPL_SDF', action='store_true')
    group.add_argument('--from_pretrained', action='store_true')
    group.add_argument('--train', action='store_true')
    group.add_argument('--validate', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--animation', action='store_true')

    parser.add_argument('--animation_pattern', default='aist_demo')
    parser.add_argument('--exp_dir', default='../exp')
    parser.add_argument('--runs_dir', default='./runs')
    parser.add_argument('--trial_name', default=None)
    parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')

    args, extras = parser.parse_known_args()
    assert not args.animation or (args.animation and args.resume_weights_only and isinstance(args.resume, str))

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.trial_name = args.trial_name or config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    config.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config')

    logger = logging.getLogger('pytorch_lightning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    pl.seed_everything(config.seed)
    torch.set_printoptions(precision=10)

    if args.pretrain_SMPL_SDF:
        # add pretrain_SMPL_SDF to ckpt_dir base dirname
        config.ckpt_dir += '_pretrain_SMPL_SDF'
    print('Storing checkpoints in ', config.ckpt_dir)

    if args.from_pretrained:
        args.train = True
        checkpoints = glob.glob(os.path.join(config.ckpt_dir + '_pretrain_SMPL_SDF', '*.ckpt'))
        print('Available checkpoints: ', [os.path.basename(ckpt) for ckpt in checkpoints])
        assert len(checkpoints) > 0, 'No checkpoints found in {}'.format(os.path.join(config.ckpt_dir + '_pretrain_SMPL_SDF', '*.ckpt'))
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Loading checkpoint: ', latest_checkpoint)
        args.resume = latest_checkpoint
        args.resume_weights_only = True
        config.system.loss.pretrained_sdf = True
        config.model.pretrained_sdf = True

    if args.validate:
        config.trainer.check_val_every_n_epoch = 1
    if args.animation:
        config.model.ray_chunk = 2 << 12

    if args.animation or args.validate or args.test:
        config.system.optimizer.args.lr = 0.
        config.trainer.max_epochs += 1

    if args.pretrain_SMPL_SDF:
        config.system.optimizer.args.lr = 0. #5e-6
        # config.system.optimizer.args.lr = 5e-3
        config.trainer.max_epochs = 30
        config.trainer.check_val_every_n_epoch = 10
        config.checkpoint.save_top_k = 1
        config.checkpoint.save_last = True
        config.model.geometry.density.params_init = {"beta": 0.075}
        datamodule = datasets.make('SMPL-SDF', config.dataset)
        system = systems.make(
            'instantgeoavatar-system-preSDF', config,
            load_from_checkpoint=None,
            datamodule=datamodule,
        )
    else:
        datamodule = datasets.make(config.dataset.name, config.dataset)
        system = systems.make(
            config.system.name, config,
            load_from_checkpoint=None if not args.resume_weights_only else args.resume,
            datamodule=datamodule,
        )

    if args.animation:
        pose_sequence_path = os.path.abspath(f"../data/animation/{args.animation_pattern}.npz")
        print('Loading animation sequence from ', pose_sequence_path)
        animation_dataset = datasets.make(
            name='animation',
            config={},
            **{'pose_sequence_path': pose_sequence_path,
             'betas': datamodule.trainset.smpl_params["betas"],
             'downscale': 2,
            }
        )
    elif not args.resume:
        print(
            'Training a system instance from scratch. '
            'You can specify a checkpoint to be resumed with --resume '
            'for finetuning the model to a set of images.'
        )

    callbacks = []
    loggers = []
    if args.train or args.pretrain_SMPL_SDF:
        callbacks += [
            ModelCheckpoint(
                dirpath=config.ckpt_dir,
                **config.checkpoint
            ),
            CodeSnapshotCallback(
                config.code_dir, use_version=False,
            ),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False,
            ),
            ModelSummary(max_depth=-1),
        ]

    if sys.platform == 'win32':
        # does not support multi-gpu on windows
        strategy = 'dp'
        assert n_gpus == 1
    else:
        strategy = 'ddp_find_unused_parameters_false'

    trainer = Trainer(
        devices=n_gpus,
        accelerator='gpu',
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        enable_checkpointing=True,
        **config.trainer
    )

    if args.train or args.pretrain_SMPL_SDF:
        if args.resume and not args.resume_weights_only:
            trainer.fit(system, datamodule=datamodule, ckpt_path=args.resume)
        else:
            trainer.fit(system, datamodule=datamodule)
    else:
        system.eval()
        if args.validate:
            trainer.validate(system, datamodule=datamodule, ckpt_path=args.resume)
        elif args.test:
            if args.resume and not args.resume_weights_only:
                trainer.test(system, datamodule=datamodule, ckpt_path=args.resume)
            else:
                trainer.test(system, datamodule=datamodule)
        elif args.animation:
            system.to(system.rank)
            system.animation(animation_dataset, args.animation_pattern)
        else:
            raise NotImplementedError('No such mode.')


if __name__ == '__main__':
    main()