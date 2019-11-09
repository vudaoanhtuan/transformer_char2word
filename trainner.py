import os

import torch
import torch.nn as nn
from tqdm import tqdm

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_padding_mask(x, padding_value=0):
    # x: BxS
    mask = x==padding_value
    return mask

class Trainer:
    def __init__(self, model, optimizer, train_dl, test_dl, weight_dir='weight', log_file='log.txt', scheduler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dl = train_dl
        self.test_dl = test_dl

        self.device = torch.device(device)

        self.log_file = log_file
        self.weight_dir = weight_dir
        if not os.path.isdir(weight_dir):
            os.mkdir(weight_dir)



    def run_iterator(self, dataloader, is_training=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        total_item = 0
        
        desc = "total_loss=%.4f | batch_loss=%.4f | lr=%.6f"
        with tqdm(total=len(dataloader)) as pbar:
            for src, tgt in dataloader:
                src = src.long()
                tgt = tgt.long()
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                tgt_inp = tgt[:, :-1]
                tgt_lbl = tgt[:, 1:]
                
                _, loss = self.model(src, tgt_inp, tgt_lbl)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        scheduler.step()
            
            total_loss += loss.item()
            total_item += 1

            pbar.update(1)
            pbar.set_description(desc%(total_loss/total_item, loss.item(), self.optimizer.param_groups[0]['lr']))
        return total_loss/total_item

    def train(self, num_epoch=10):
        for e in range(num_epoch):
            print('\n[Epoch %d/%d] ========\n' % (e, num_epoch) ,flush=True, end='')
            train_loss = self.run_iterator(self.train_dl)
            val_loss = self.run_iterator(self.test_dl, is_training=False)
            torch.save(self.model.state_dict(), os.path.join(self.weight_dir, 'model.%02d.h5'%e))

