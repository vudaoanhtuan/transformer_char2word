import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

import pandas as pd
from tqdm import tqdm

from utils.preprocess_utils import pad_sequences


class Dataset(data.Dataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=50):
        self.tokenizer = tokenizer

        df = pd.read_csv(file_path, sep='\t', names=['src', 'tgt'])

        tokens = [tokenizer.tokenize(str(x.src), str(x.tgt)) for i, x in tqdm(df.iterrows())]
        self.src = [x[0] for x in tokens]
        self.tgt = [x[1] for x in tokens]

        self.src = pad_sequences(self.src, maxlen=src_pad_len, value=tokenizer.pad, padding='post')
        self.tgt = pad_sequences(self.tgt, maxlen=tgt_pad_len, value=tokenizer.pad, padding='post')

    def regenerate_source(self):
        pass

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]
        tgt_inp = tgt[:-1]
        tgt_lbl = tgt[1:]
        return src, tgt_inp, tgt_lbl

class MaskDataset(data.Dataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=50, use_mask=True):
        self.tokenizer = tokenizer
        self.use_mask = use_mask
        self.src_pad_len = src_pad_len
        self.tgt_pad_len = tgt_pad_len

        df = pd.read_csv(file_path, sep='\t', names=['src', 'tgt'])

        tokens = [tokenizer.tokenize(str(x.src), str(x.tgt)) for i, x in tqdm(df.iterrows())]
        self.src = [x[0] for x in tokens]
        self.tgt = [x[1] for x in tokens]

        self.pad_value = self.tokenizer.pad
        self.mask_value = self.tokenizer.mask

        self.src_vocab_len = len(self.tokenizer.src_stoi)
        self.tgt_vocab_len = len(self.tokenizer.tgt_stoi)

    def mask_item(self, x,
        pad_value, mask_value, vocab_len,
        pc_choice=0.15, pc_mask=0.8, pc_other=0.1, pc_keep=0.1):
        n_mask = int(pc_choice*len(x))

        if n_mask > 0:
            mask_ix = np.random.choice(range(0, len(x)), n_mask)
            mask_v = np.random.choice([0,1,2], n_mask, p=[pc_mask, pc_other, pc_keep])

            x = x.copy()
            for i,v in zip(mask_ix, mask_v):
                if v==0:
                    x[i] = mask_value
                elif v==1:
                    x[i] = np.random.randint(5, vocab_len)
            return x

        return x

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = self.src[index]
        tgt_inp = self.tgt[index][:-1]
        tgt_lbl = self.tgt[index][1:]

        if self.use_mask:
            src = self.mask_item(src, self.pad_value, self.mask_value, self.src_vocab_len)
            tgt_inp = self.mask_item(tgt_inp, self.pad_value, self.mask_value, self.tgt_vocab_len)

        src = pad_sequences([src], maxlen=self.src_pad_len, value=self.tokenizer.pad, padding='post')[0]
        tgt_inp = pad_sequences([tgt_inp], maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')[0]
        tgt_lbl = pad_sequences([tgt_lbl], maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')[0]

        return src, tgt_inp, tgt_lbl

