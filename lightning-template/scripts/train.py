import sys
sys.path.append('.')

import os
from distutils.command.config import config
from pathlib import Path
import logging
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from seg_det.util import setup_config, setup_experiment

log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'base_config.yaml'


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)
    
    pl.seed_everything(cfg.experiment.seed, workers=True)
    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)


    #Create and load model / data
    model_module, data_module = setup_experiment(cfg)
    
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project, 
                                   save_dir=cfg.experiment.save_dir,
                                   id = cfg.experiment.uuid)
    
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                            every_n_train_steps=cfg.experiment.checkpoint_interval),
    ]
    
    #Train
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         **cfg.trainer)

    trainer.fit(model_module, datamodule=data_module)
    
    
if __name__ == '__main__':
    main()