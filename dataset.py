from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
from tqdm import tqdm

from utils.preprocess_utils import pad_sequences
from utils.word_transform import transform_sentence


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
        self.num_worker = cpu_count()

        with open(file_path) as f:
            self.corpus = f.read().split('\n')[:-1]

        self.src = None
        self.tgt = [tokenizer.tokenize_tgt(x) for x in tqdm(self.corpus)]

        self.src_vocab_len = len(self.tokenizer.src_stoi)
        self.tgt_vocab_len = len(self.tokenizer.tgt_stoi)
        self.src_pad_len = src_pad_len
        self.tgt_pad_len = tgt_pad_len

        self.regenerate_source()

    def random_mask_tgt(self, x, pc_mask=0.3):
        mask = np.random.choice([0,1], len(x), p=[1-pc_mask, pc_mask])
        x = np.where(mask==1, self.tokenizer.mask, x)
        return x

    def regenerate_source(self):
        del self.src
        with Pool(self.num_worker) as p:
            self.src = list(tqdm(p.imap(transform_sentence, self.corpus), total=len(self.corpus)))
        self.src = [self.tokenizer.tokenize_src(x) for x in tqdm(self.src)]


    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        src = self.src[index]
        tgt_inp = self.tgt[index][:-1]
        tgt_lbl = self.tgt[index][1:]

        if self.use_mask:
            tgt_inp = self.random_mask_tgt(tgt_inp)

        src = pad_sequences([src], maxlen=self.src_pad_len, value=self.tokenizer.pad, padding='post')[0]
        tgt_inp = pad_sequences([tgt_inp], maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')[0]
        tgt_lbl = pad_sequences([tgt_lbl], maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')[0]

        return src, tgt_inp, tgt_lbl

