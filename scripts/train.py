import os
import sys

sys.path.append('.')
sys.path.append('..')

from omegaconf import OmegaConf
from dataset import dataset
from training.coach import Coach
import utils
import os
import logging
import wandb
from dataset import factory


logging.basicConfig(
    level=logging.INFO,
    format="INFO: %(message)s"
    )

def main():
    utils.torch_seed(cfg.SEED)
    if cfg.TRAIN.use_wandb:
        wandb.init(name=cfg.EXP_NAME, project=cfg.MODEL.model_name, config=OmegaConf.to_container(cfg))

    # download the dataset
    if not os.path.exists(cfg.DATASET.datadir):
        print("Downloading the study dataset ...")
        dataset.download_dataset(cfg.DATASET.url, cfg.DATASET.datadir)
    
    # train, val split
    check_val_files = [file for file in os.listdir(cfg.DATASET.datadir) if 'val' in file]
    if not check_val_files:
        factory.train_val_split(cfg.DATASET.datadir, cfg.DATASET.val_ratio)

    # select the coach
    coach = Coach(cfg)
    coach.train()

if __name__=='__main__':
    args = OmegaConf.from_cli()

    # load default config
    total_cfg = OmegaConf.load(args.configs)
    del args['configs']
    for exp_name, cfg in total_cfg['EXPERIMENTS'].items():
        print(f"The experiments '{exp_name}' starts !!")
        print(OmegaConf.to_yaml(cfg))
        main()

