import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import spacy

from keras.preprocessing.sequence import pad_sequences

class Tokenizer:
    def __init__(self, src_vocab, tgt_vocab, lower_text=True):
        disable = ['vectors', 'textcat', 'tagger', 'parser', 'ner']
        self.word_tokenizer = spacy.load('en_core_web_sm', disable=disable)
        self.lower_text = lower_text
        self.special = ['<pad>', '<unk>', '<bos>', '<eos>']
        src_vocab = self.special + src_vocab
        tgt_vocab = self.special + tgt_vocab
        self.pad = 0
        self.unk = 1
        self.bos = 2
        self.eos = 3
        self.src_itos = {i:s for i,s in enumerate(src_vocab)}
        self.tgt_itos = {i:s for i,s in enumerate(tgt_vocab)}
        self.src_stoi = {s:i for i,s in enumerate(src_vocab)}
        self.tgt_stoi = {s:i for i,s in enumerate(tgt_vocab)}
        assert self.src_stoi['<pad>'] == self.tgt_stoi['<pad>'] == self.pad
        assert self.src_stoi['<unk>'] == self.tgt_stoi['<unk>'] == self.unk
        assert self.src_stoi['<bos>'] == self.tgt_stoi['<bos>'] == self.bos
        assert self.src_stoi['<eos>'] == self.tgt_stoi['<eos>'] == self.eos
        

    def _char_tokenize(self, sent):
        if self.lower_text:
            sent = sent.lower()
        return [c for c in sent]

    def _word_tokenize(self, sent):
        if self.lower_text:
            sent = sent.lower()
        words = self.word_tokenizer(sent)
        words = [w.text for w in words]
        return words

    def _token_to_id(self, tokens, stoi):
        idxs = []
        for t in tokens:
            idx = stoi.get(t)
            idx = idx if idx else self.unk
            idxs.append(idx)
        return idxs

    def _id_to_token(self, idxs, itos):
        tokens = []
        for i in idxs:
            t = itos.get(i)
            t = t if t else self.special[self.unk]
            tokens.append(t)
        return tokens

    def tokenize(self, src, tgt):
        src = self._char_tokenize(src)
        src = self._token_to_id(src, self.src_stoi)

        tgt = self._word_tokenize(tgt)
        tgt = self._token_to_id(tgt, self.tgt_stoi)
        tgt = [self.bos] + tgt + [self.eos]

        return src, tgt


def load_tokenizer(src_vocab_path, tgt_vocab_path):
    with open(src_vocab_path) as f:
        src_vocab = f.read().split('\n')
        if len(src_vocab[-1]) == 0:
            src_vocab = src_vocab[:-1]
    
    with open(tgt_vocab_path) as f:
        tgt_vocab = f.read().split('\n')
        if len(tgt_vocab[-1]) == 0:
            tgt_vocab = tgt_vocab[:-1]

    tokenizer = Tokenizer(src_vocab, tgt_vocab)

    return tokenizer


class Dataset(data.Dataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=100):
        self.tokenizer = tokenizer

        with open(file_path) as f:
            corpus = f.read().split('\n')[:-1]
        corpus = [x.split('\t') for x in corpus]

        tokens = [tokenizer.tokenize(*x) for x in corpus]
        self.src = [x[0] for x in tokens]
        self.tgt = [x[1] for x in tokens]

        self.src = pad_sequences(self.src, maxlen=src_pad_len, value=tokenizer.pad, padding='post')
        self.tgt = pad_sequences(self.tgt, maxlen=tgt_pad_len, value=tokenizer.pad, padding='post')


    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]
        return src, tgt


if __name__ == "__main__":
    tokenizer = load_tokenizer('data/src_vocab.txt', 'data/tgt_vocab.txt')
    ds = Dataset('data/dump.tsv', tokenizer, src_pad_len=10, tgt_pad_len=5)
    dl = data.DataLoader(ds, batch_size=3, shuffle=True)
    for src, tgt in dl:
        print(src)
        print(tgt)
        print()


