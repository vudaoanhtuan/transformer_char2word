from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
from tqdm import tqdm

from utils.preprocess_utils import pad_sequences
from utils.word_transform import transform_sentence

class SingleDataset(data.Dataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=50):
        self.tokenizer = tokenizer
        self.src_pad_len = src_pad_len
        self.tgt_pad_len = tgt_pad_len

        with open(file_path) as f:
            self.corpus = f.read().split("\n")[:-1]

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        sent = self.corpus[index]

        tgt_token = self.tokenizer.tokenize_tgt(sent)
        tgt_inp = tgt_token[:-1]
        tgt_lbl = tgt_token[1:]

        sent = transform_sentence(sent)
        src = self.tokenizer.tokenize_src(sent)

        src = pad_sequences([src], maxlen=self.src_pad_len, value=self.tokenizer.pad, padding='post')[0]
        tgt_inp = pad_sequences([tgt_inp], maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')[0]
        tgt_lbl = pad_sequences([tgt_lbl], maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')[0]

        return src, tgt_inp, tgt_lbl
