import os

import torch
import torch.nn as nn
from tqdm import tqdm

from logger import Logger

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_padding_mask(x, padding_value=0):
    # x: BxS
    mask = x==padding_value
    return mask

class Trainer:
    def __init__(self, model, optimizer, train_dl, test_dl, weight_dir='weight', log_dir='logs', scheduler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dl = train_dl
        self.test_dl = test_dl

        self.device = torch.device(device)
        self.model.to(self.device)

        self.weight_dir = weight_dir
        if not os.path.isdir(weight_dir):
            os.mkdir(weight_dir)

        if os.path.isdir(log_dir):
            import shutil
            shutil.rmtree(log_dir)

        self.logger = Logger(log_dir)
        self.train_step = 0


    def run_iterator(self, dataloader, is_training=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        total_item = 0

        desc = "total_loss=%.6f | batch_loss=%.6f | lr=%.6f"
        with tqdm(total=len(dataloader)) as pbar:
            for src, tgt_inp, tgt_lbl in dataloader:
                src = src.long().to(self.device)
                tgt_inp = tgt_inp.long().to(self.device)
                tgt_lbl = tgt_lbl.long().to(self.device)
                
                self.optimizer.zero_grad()
                _, loss = self.model(src, tgt_inp, tgt_lbl)

                if is_training:
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item()
                total_item += 1

                pbar.update(1)
                pbar.set_description(desc%(total_loss/total_item, loss.item(), self.optimizer.param_groups[0]['lr']))

                if is_training:
                    info = {"train_loss": loss.item()}
                    self.train_step += 1
                    step = self.train_step
                    self.logger.update_step(info, step)

        return total_loss/total_item

    def train(self, num_epoch=10):
        for epoch in range(num_epoch):
            print('\n[Epoch %d/%d] ========\n' % (epoch, num_epoch) ,flush=True, end='')
            train_loss = self.run_iterator(self.train_dl)
            val_loss = self.run_iterator(self.test_dl, is_training=False)
            torch.save(self.model.state_dict(), os.path.join(self.weight_dir, 'model.%02d.h5'%epoch))

            losses = {
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            self.logger.update_epoch(losses, epoch)



class MaskTrainer(Trainer):
    def run_iterator(self, dataloader, is_training=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        total_item = 0

        desc = "total_loss=%.6f | batch_loss=%.6f | lr=%.6f"
        with tqdm(total=len(dataloader)) as pbar:
            for src, src_mask, tgt_inp, tgt_lbl, tgt_mask in dataloader:
                src = src.long().to(self.device)
                src_mask = src_mask.long().to(self.device)
                tgt_inp = tgt_inp.long().to(self.device)
                tgt_lbl = tgt_lbl.long().to(self.device)
                tgt_mask = tgt_mask.long().to(self.device)
                
                self.optimizer.zero_grad()
                _, loss = self.model(src, tgt_inp, tgt_lbl)

                if is_training:
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item()
                total_item += 1

                pbar.update(1)
                pbar.set_description(desc%(total_loss/total_item, loss.item(), self.optimizer.param_groups[0]['lr']))

                if is_training:
                    info = {"train_loss": loss.item()}
                    self.train_step += 1
                    step = self.train_step
                    self.logger.update_step(info, step)

        return total_loss/total_item