import os
import logging
import json
import wandb
import numpy as np
from omegaconf import OmegaConf
from study import utils
from dataset import factory
import metrics

import torch
import torch.nn.functional as F

_logger = logging.getLogger('train')

class AverageMeter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

class Coach():
    def __init__(self, cfg):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        _logger.info(f"[DEVICE]: {self.device}")

        self.cfg = cfg

        # Make the directory for the results
        self.savedir = os.path.join(cfg.RESULT.savedir, cfg.MODEL.model_name)
        os.makedirs(self.savedir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.savedir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        kwargs = {"pyramid": cfg.MODEL.pyramid, "freeze": cfg.TRAIN.freeze} if cfg.MODEL.model_name == "resnet50_pretrained" else {}
        self.model = utils.choose_model(cfg.MODEL.model_name, **kwargs).to(self.device)
        self.optimizer = utils.choose_optimizer(        
            name = cfg.OPTIMIZER.name,
            model = self.model,
            lr = cfg.OPTIMIZER.lr,
            betas=cfg.OPTIMIZER.betas,
            weight_decay=cfg.OPTIMIZER.weight_decay
        )

        # loss function
        self.criterion = utils.choose_criterion(cfg.TRAIN.transform)

        # scheduler
        if cfg.SCHEDULER.scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.TRAIN.max_step/cfg.SCHEDULER.cycle)
        
        # Split and load train and test dataset
        self.trainset = factory.create_dataset(cfg.DATASET.datadir, 'train', cfg.MODEL.model_name, cfg.TRAIN.transform)
        self.valset = factory.create_dataset(cfg.DATASET.datadir, 'val', cfg.MODEL.model_name, cfg.VAL.transform)
        self.train_loader = factory.creat_dataloader(self.trainset, cfg.DATALOADER.batch_size, cfg.DATALOADER.shuffle)
        self.val_loader = factory.creat_dataloader(self.valset, cfg.DATALOADER.batch_size, cfg.DATALOADER.shuffle)
    
    def train(self):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()

        self.model.train()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        no_developed = 0
        best =  0
        step = 1
        train_flag = True
        while train_flag:
            for inputs, targets in self.train_loader:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                pred = self.model(inputs)

                if self.cfg.TRAIN.transform == 'softcrop':
                    log_probs = F.log_softmax(pred, dim=1)
                    loss = self.criterion(log_probs, targets, reduction='batchmean')
                else:
                    loss = self.criterion(pred, targets)
                
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                losses_m.update(loss.item())

                # wandb
                if self.cfg.TRAIN.use_wandb:
                    wandb.log({
                        'lr':self.optimizer.param_groups[0]['lr'],
                        'train_loss':losses_m.val
                    },
                    step=step)

                # show the log
                if step % self.cfg.LOG.log_interval == 0:
                    _logger.info(f"Step-{step}, batch_time-{batch_time_m.avg:.2f} ms, "
                                 f"loss-{losses_m.avg:.2f}, memory-{torch.cuda.memory_reserved(self.device)/(1000**3):.2f}GiB "
                                 f"lr-{self.optimizer.param_groups[0]['lr']:.6f}"
                    )

                # evaluation
                if step % self.cfg.TRAIN.eval_interval == 0 or step==1:
                    results = self.validate()
                    curr_f1 = results['f1 score']
                    curr_top1_acc = results['Top1']
                    _logger.info(f"[EVAL] Step-{step}, top1 accuracy-{curr_top1_acc*100:.2f}%, f1 score-{curr_f1:.2f}")

                    if curr_f1 < best:
                        no_developed += 1

                    self.model.train()
                    eval_log = {f'eval_{k}': v for k, v in results.items()}

                    # wandb
                    if self.cfg.TRAIN.use_wandb:
                        wandb.log(eval_log, step=step)

                    # check for the model which has best top1 accuracy
                    if curr_f1 > best:
                        state = {'best_top1_step': step}
                        state.update(eval_log)
                        state.update(OmegaConf.to_container(self.cfg))

                        file_path = os.path.join(self.savedir, f"{self.cfg.EXP_NAME}.json")
                        with open(file_path, "w") as f:
                            json.dump(state, f, indent=4)

                        torch.save(self.model.state_dict(), os.path.join(self.savedir, f"{self.cfg.EXP_NAME}.pt"))

                        _logger.info(f"[BEST!] Step-{step}, delta for the best-{no_developed} steps, Best f1_score: {best:.2f} to {curr_f1:.2f}")
                        best = curr_f1
                        no_developed = 0

                # save the latest model
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'latest_model.pt'))

                if self.cfg.SCHEDULER.scheduler:
                    self.scheduler.step()
                
                # check the time for each batch
                torch.cuda.synchronize()    # make the all the workings in gpus done
                end.record()                # mark the current status in gpus
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
                if step != 1:
                    batch_time_m.update(elapsed_time)

                # stop training
                if step == self.cfg.TRAIN.max_step:
                    train_flag=False
                    break
                step += 1

    def validate(self):
        self.model.eval()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                pred = self.model(inputs)

                all_preds.append(pred.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            top1_acc = metrics.top1_accuracy(all_preds, all_targets, num_classes=10, average='macro')
            f1_score = metrics.f1_score(all_preds, all_targets, num_classes=10, average='macro')

            results = {
                "Top1": top1_acc,
                "f1 score": f1_score
            }
        return results
