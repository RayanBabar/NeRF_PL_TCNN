import os, sys
from opt import opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

#models
from models.NeRF import Embedding, NeRF
from models.rendering import render_rays

#optimizer, schedular, visualization
from utils import *

#losses
from losses import loss_dict

#metrics
from metrics import *

#pytorch lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import torch.multiprocessing as mp

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays']
        rgbs = batch['rgbs']
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = render_rays(
                self.models,
                self.embeddings,
                rays[i:i+self.hparams.chunk],
                self.hparams.N_samples,
                self.hparams.use_disp,
                self.hparams.perturb,
                self.hparams.noise_std,
                self.hparams.N_importance,
                self.hparams.chunk,
                self.train_dataset.white_back
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}

        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        schedular = get_scheduler(self.hparams, self.optimizer)

        return [self.optimizer], [schedular]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)

        # Compute loss
        train_loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        # Compute PSNR
        with torch.inference_mode():
            train_psnr = psnr(results[f'rgb_{typ}'], rgbs)

        # Log training loss and PSNR for checkpoint monitoring
        self.log('train/loss', train_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log('train/psnr', train_psnr, prog_bar=True, logger=True, sync_dist=True)

        # Log learning rate
        current_lr = get_learning_rate(self.optimizer)
        self.log('lr', current_lr, prog_bar=True, logger=True, sync_dist=True)

        return {'loss': train_loss, 'train_psnr': train_psnr}


    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze()
        rgbs = rgbs.squeeze()
        results = self(rays)

        val_loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))
            stack = torch.stack([img_gt, img, depth])
            self.logger.experiment.add_images('val/GT_pred_depth', stack, self.global_step)

        val_psnr = psnr(results[f'rgb_{typ}'], rgbs)

        # Log validation loss and PSNR for checkpoint monitoring
        self.log('val/loss', val_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/psnr', val_psnr, prog_bar=True, logger=True, sync_dist=True)

        return {'val_loss': val_loss, 'val_psnr': val_psnr}

    def on_validation_epoch_end(self, outputs=None):
        if outputs:
            # Aggregate the losses and PSNR values from all validation steps
            mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

            # Log aggregated values for monitoring
            self.log('val/loss', mean_loss, prog_bar=True, logger=True, sync_dist=True)
            self.log('val/psnr', mean_psnr, prog_bar=True, logger=True, sync_dist=True)

            return {
                'progress_bar': {'val_loss': mean_loss, 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss, 'val/psnr': mean_psnr}
            }



if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    args = opts()
    system = NeRFSystem(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(f'ckpts/{args.exp_name}'),
        filename='{epoch:d}',
        monitor='val/loss',
        mode='min',
        save_top_k=5
    )

    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.exp_name,
        log_graph=False,  # If you want to log the computation graph, set to True
        default_hp_metric=False  # To avoid logging the default hp_metric
    )

    trainer = Trainer(max_epochs=args.num_epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      enable_model_summary=True,
                      enable_progress_bar=True,
                      devices=args.num_gpus,
                      accelerator="gpu" if args.num_gpus > 0 else "cpu",
                      strategy='ddp' if args.num_gpus > 1 else 'auto',
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="advanced" if args.num_gpus == 1 else None)

    # Fit model and resume from checkpoint if necessary
    if args.ckpt_path is not None:
        trainer.fit(system, ckpt_path=args.ckpt_path)  # Resume training from checkpoint
    else:
        trainer.fit(system)  # Start training normally
