from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
from tqdm import tqdm

from utils.preprocess_utils import pad_sequences
from utils.word_transform import transform_sentence
from utils.teencode import transform_teencode

class SingleDataset(data.Dataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=50, seed=None):
        self.tokenizer = tokenizer
        self.num_worker = cpu_count()

        self.src_pad_len = src_pad_len
        self.tgt_pad_len = tgt_pad_len

        with open(file_path) as f:
            self.corpus = f.read().split('\n')[:-1]

        self.src = None
        self.tgt = [tokenizer.tokenize_tgt(x) for x in tqdm(self.corpus)]
        self.tgt = pad_sequences(self.tgt, maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')

        self.src_vocab_len = len(self.tokenizer.src_stoi)
        self.tgt_vocab_len = len(self.tokenizer.tgt_stoi)

        if seed:
            np.random.seed(seed)
        self.regenerate_source()

    def regenerate_source(self):
        del self.src
        with Pool(self.num_worker) as p:
            self.src = list(tqdm(p.imap(transform_sentence, self.corpus), total=len(self.corpus)))
        self.src = [self.tokenizer.tokenize_src(x) for x in tqdm(self.src)]
        self.src = pad_sequences(self.src, maxlen=self.src_pad_len, value=self.tokenizer.pad, padding='post')


    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        src = self.src[index]
        tgt_inp = self.tgt[index][:-1]
        tgt_lbl = self.tgt[index][1:]

        return src, tgt_inp, tgt_lbl