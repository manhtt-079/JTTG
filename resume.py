from pytorch_lightning import Trainer, seed_everything
from lightning_trainer import ExAbModel
from loguru import logger
import pandas as pd
import argparse
import gc
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config.config import Config

def main(config: Config, task_name: str):
    
    wandb_logger = WandbLogger(project=task_name, save_dir=config.trainer_args.log)
    early_stopping = EarlyStopping(
        monitor=config.trainer_args.monitor,
        mode='min',
        min_delta=config.trainer_args.delta,
        patience=config.trainer_args.patience,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.trainer_args.checkpoint,
        filename=config.model_name+'-{epoch}-{step}-{val_loss:.2f}',
        monitor=config.trainer_args.monitor,
        mode='min',
        save_top_k=config.trainer_args.save_top_k,
        save_on_train_epoch_end=config.trainer_args.save_on_train_epoch_end
    )
    
    trainer = Trainer(
        resume_from_checkpoint='/home/int2-user/project_sum/checkpoint/vit5-sum-vnds/vit5-sum-epoch=19-step=15500-val_loss=2.50.ckpt',
        accelerator=config.trainer_args.accelerator,
        accumulate_grad_batches=config.trainer_args.accumulate_grad_batches,
        amp_backend=config.trainer_args.amp_backend,
        auto_lr_find=config.trainer_args.auto_lr_find,
        auto_scale_batch_size=config.trainer_args.auto_scale_batch_size,
        auto_select_gpus=config.trainer_args.auto_select_gpus,
        callbacks=[early_stopping, checkpoint_callback],
        default_root_dir=config.trainer_args.checkpoint,
        enable_model_summary=config.trainer_args.enable_model_summary,
        enable_checkpointing=config.trainer_args.enable_checkpointing,
        max_epochs=config.trainer_args.max_epochs + 10,
        logger=wandb_logger,
        log_every_n_steps=config.trainer_args.eval_steps,
        precision=config.trainer_args.precision
    )
    
    gc.collect()
    model = ExAbModel(config)
    trainer.fit(model)
    
    logger.info('----- Testing -----')
    predictions = trainer.predict(dataloaders=model.test_dataloader(), ckpt_path='best')
    rouge_scores = pd.DataFrame(predictions).mean().to_dict()
    logger.info(rouge_scores)

 
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='The random seed for reproducibility.')
    parser.add_argument('--task', type=str, default='task1')
    parser.add_argument('--config_file', type=str, default='./config/config.ini', help='The configuration file.')
    
    args = parser.parse_args()
    seed_everything(args.seed)
    EXPERIMENT_MAP = {
        'task1': {
            'dataset_name': 'reddit_tifu',
            'model_name': 'bart-sum',
            'is_long': True
        },
        'task2': {
            'dataset_name': 'bill_sum',
            'model_name': 'bart-sum',
            'use_us_test': True
        },
        'task3': {
            'dataset_name': 'reddit_tifu',
            'model_name': 't5-sum',
            'is_long': True
        },
        'task4': {
            'dataset_name': 'bill_sum',
            'model_name': 't5-sum',
            'use_us_test': True
        },
        'task5': {
            'dataset_name': 'vnds',
            'model_name': 'bartpho-sum'
        },
        'task6': {
            'dataset_name': 'vnds',
            'model_name': 'vit5-sum'
        }
    }

    kwargs = EXPERIMENT_MAP[args.task]
    config = Config(config_file=args.config_file, **kwargs)
    main(config=config)