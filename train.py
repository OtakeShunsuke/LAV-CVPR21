import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import losses
import wandb

import os
import numpy as np

import utils
import align_dataset
from models import BaseModel, ConvEmbedder
from config import CONFIG

# from ViViT import ViViT
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce, repeat
from IPython.display import display
import argparse


class AlignNet(LightningModule):
    def __init__(self, config):
        super(AlignNet, self).__init__()

        self.base_cnn = BaseModel(pretrained=True)

        if config.TRAIN.FREEZE_BASE:
            if config.TRAIN.FREEZE_BN_ONLY:
                utils.freeze_bn_only(module=self.base_cnn)
            else:
                utils.freeze(module=self.base_cnn, train_bn=False)

        self.emb = ConvEmbedder(
            emb_size=config.DTWALIGNMENT.EMBEDDING_SIZE,
            l2_normalize=config.LOSSES.L2_NORMALIZE,
        )

        self.lav_loss = losses.LAV(
            alpha=config.LOSSES.ALPHA,
            sigma=config.LOSSES.SIGMA,
            margin=config.LOSSES.IDM_IDX_MARGIN,
            num_frames=config.TRAIN.NUM_FRAMES,
            dtw_gamma=config.DTWALIGNMENT.SDTW_GAMMA,
            dtw_normalize=config.DTWALIGNMENT.SDTW_NORMALIZE,
            debug=False,
        )

        self.description = config.DESCRIPTION

        # params
        self.l2_normalize = config.LOSSES.L2_NORMALIZE
        self.alpha = config.LOSSES.ALPHA
        self.sigma = config.LOSSES.SIGMA

        self.lr = config.TRAIN.LR
        self.weight_decay = config.TRAIN.WEIGHT_DECAY
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.freeze_base = config.TRAIN.FREEZE_BASE
        self.freeze_bn_only = config.TRAIN.FREEZE_BN_ONLY

        self.data_path = os.path.abspath(config.DATA_PATH)

        # PennActionデータセットのフォルダ構成のために追加---------------------------------
        self.target_pennaction = config.PENN_ACTION.NAME
        # -----------------------------------------------------------------------------

        self.hparams.config = config

        # self.save_hyperparameters()

    def train(self, mode=True):
        super(AlignNet, self).train(mode=mode)

        if self.freeze_base:
            if self.freeze_bn_only:
                utils.freeze_bn_only(module=self.base_cnn)
            else:
                utils.freeze(module=self.base_cnn, train_bn=False)

    def forward(self, x):
        num_ctxt = self.hparams.config.DATA.NUM_CONTEXT

        num_frames = x.size(1) // num_ctxt
        x = self.base_cnn(x)
        x = self.emb(x, num_frames)
        return x

    def training_step(self, batch, batch_idx):
        print("batch_idx: {}".format(batch_idx))
        (a_X, _, a_steps, a_seq_len), (b_X, _, b_steps, b_seq_len) = batch

        X = torch.cat([a_X, b_X])
        embs = self.forward(X)
        a_embs, b_embs = torch.split(embs, a_X.size(0), dim=0)

        loss = 0.0

        for a_emb, a_idx, a_len, b_emb, b_idx, b_len in zip(
            a_embs.unsqueeze(1),
            a_steps,
            a_seq_len,
            b_embs.unsqueeze(1),
            b_steps,
            b_seq_len,
        ):
            loss += self.lav_loss(a_emb, b_emb, a_idx, b_idx, a_len, b_len)

        loss = loss / self.batch_size

        tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        (a_X, _, a_steps, a_seq_len), (b_X, _, b_steps, b_seq_len) = batch

        X = torch.cat([a_X, b_X])
        embs = self.forward(X)
        a_embs, b_embs = torch.split(embs, a_X.size(0), dim=0)

        loss = 0.0

        for a_emb, a_idx, a_len, b_emb, b_idx, b_len in zip(
            a_embs.unsqueeze(1),
            a_steps,
            a_seq_len,
            b_embs.unsqueeze(1),
            b_steps,
            b_seq_len,
        ):
            loss += self.lav_loss(
                a_emb, b_emb, a_idx, b_idx, a_len, b_len, logger=self.logger
            )

        loss = loss / self.batch_size

        tensorboard_logs = {"val_loss": loss}
        self.log("val_loss", loss)

        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {}

        for x in outputs:
            for k in x["log"]:
                if k not in tensorboard_logs:
                    tensorboard_logs[k] = []

                tensorboard_logs[k].append(x["log"][k])

        for k, losses in tensorboard_logs.items():
            tensorboard_logs[k] = torch.stack(losses).mean()

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def train_dataloader(self):
        config = self.hparams.config
        train_path = os.path.join(self.data_path, "train")

        # PennActionデータセットのフォルダ構成のために追加---------------------------------
        if not self.target_pennaction == None:
            train_path = os.path.join(train_path, self.target_pennaction)
        # ---------------------------------------------------------------------------

        train_transforms = utils.get_transforms(augment=True)
        data = align_dataset.AlignData(
            train_path,
            config.TRAIN.NUM_FRAMES,
            config.DATA,
            transform=train_transforms,
            flatten=False,
        )
        data_loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.DATA.WORKERS,
        )

        return data_loader

    def val_dataloader(self):
        config = self.hparams.config
        val_path = os.path.join(self.data_path, "val")

        # PennActionデータセットのフォルダ構成のために追加---------------------------------
        if not self.target_pennaction == None:
            val_path = os.path.join(val_path, self.target_pennaction)
        # ---------------------------------------------------------------------------

        val_transforms = utils.get_transforms(augment=False)
        data = align_dataset.AlignData(
            val_path,
            config.EVAL.NUM_FRAMES,
            config.DATA,
            transform=val_transforms,
            flatten=False,
        )
        data_loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.DATA.WORKERS,
        )

        return data_loader

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


def main(hparams):
    seed_everything(hparams.SEED)

    model = AlignNet(hparams)

    dd_backend = None
    # if hparams.GPUS < 0 or hparams.GPUS > 1:
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     dd_backend = 'ddp'

    # PennActionデータセットのフォルダ構成のために追加---------------------------------
    now = datetime.datetime.now()
    if hparams.TRAIN.DEBUG:
        file_tiemstamp = "debug"
    else:
        file_tiemstamp = now.strftime("%Y%m%d%H%M%S")
    if not hparams.PENN_ACTION.NAME == None:
        root_path = os.path.join(hparams.ROOT, hparams.PENN_ACTION.NAME)
        root_path = os.path.join(root_path, file_tiemstamp)
        wandb_logger = WandbLogger(
            project="LAV",
            name="LAV_PennAction={}_{}".format(
                hparams.PENN_ACTION.NAME, file_tiemstamp
            ),
            log_model="all",
        )
    else:
        root_path = os.path.join(hparams.ROOT, file_tiemstamp)
        wandb_logger = WandbLogger(project="LAV", name="LAV_{}".format(file_tiemstamp))
    # ---------------------------------------------------------------------------

    checkpoint_path = os.path.join(root_path, "STEPS")

    try:
        checkpoint_callback = utils.CheckpointEveryNSteps(
            hparams.TRAIN.SAVE_INTERVAL_ITERS,
            filepath=checkpoint_path,
        )

        trainer = Trainer(
            gpus=hparams.GPUS,
            max_epochs=hparams.TRAIN.EPOCHS,
            # default_root_dir=hparams.ROOT,
            default_root_dir=root_path,
            deterministic=True,
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=1,
            logger=wandb_logger,
            # fast_dev_run=hparams.TRAIN.DEBUG,
        )
        wandb_logger.experiment.config.update(hparams)
        print(hparams)
        wandb_logger.watch(model)
        trainer.fit(model)
        #  distributed_backend=dd_backend, row_log_interval=10 limit_val_batches=hparams.TRAIN.VAL_PERCENT
    except KeyboardInterrupt:
        pass
    finally:
        if not hparams.TRAIN.DEBUG:
            trainer.save_checkpoint(
                os.path.join(
                    checkpoint_path,
                    "final_model_l2norm-{}"
                    "_sigma-{}_alpha-{}"
                    "_lr-{}_bs-{}.pth".format(
                        hparams.LOSSES.L2_NORMALIZE,
                        hparams.LOSSES.SIGMA,
                        hparams.LOSSES.ALPHA,
                        hparams.TRAIN.LR,
                        hparams.TRAIN.BATCH_SIZE,
                    ),
                )
            )
            trainer.save_checkpoint(
                os.path.join(
                    hparams.ROOT,
                    "final_model_l2norm-{}"
                    "_sigma-{}_alpha-{}"
                    "_lr-{}_bs-{}.pth".format(
                        hparams.LOSSES.L2_NORMALIZE,
                        hparams.LOSSES.SIGMA,
                        hparams.LOSSES.ALPHA,
                        hparams.TRAIN.LR,
                        hparams.TRAIN.BATCH_SIZE,
                    ),
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="Description of the experiment run!",
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to save checkpoints"
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset")
    parser.add_argument("--num_frames", type=int, default=None, help="Path to dataset")
    parser.add_argument("--workers", type=int, default=30, help="Path to dataset")
    parser.add_argument(
        "--action_name", type=str, default=None, help="Select action name"
    )
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")

    args = parser.parse_args()

    CONFIG.DESCRIPTION = args.description
    CONFIG.GPUS = args.gpus

    if args.root_dir:
        CONFIG.ROOT = args.root_dir
    if args.ckpt_path:
        CONFIG.CKPT_PATH = args.ckpt_path
    if args.data_path:
        CONFIG.DATA_PATH = args.data_path
    if args.num_frames:
        CONFIG.TRAIN.NUM_FRAMES = args.num_frames
        CONFIG.EVAL.NUM_FRAMES = args.num_frames
    if args.workers:
        CONFIG.DATA.WORKERS = args.workers
    if args.action_name:
        CONFIG.PENN_ACTION.NAME = args.action_name
    if args.debug:
        CONFIG.TRAIN.DEBUG = args.debug

    main(CONFIG)
