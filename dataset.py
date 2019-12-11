import six
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

import pandas as pd
import spacy

from tqdm import tqdm

class Tokenizer:
    def __init__(self, src_vocab, tgt_vocab, lower_text=True):
        disable = ['vectors', 'textcat', 'tagger', 'parser', 'ner']
        self.word_tokenizer = spacy.load('en_core_web_sm', disable=disable)
        self.lower_text = lower_text
        self.special = ['[pad]', '[unk]', '[bos]', '[eos]', '[mask]']
        src_vocab = self.special + src_vocab
        tgt_vocab = self.special + tgt_vocab
        self.pad = 0
        self.unk = 1
        self.bos = 2
        self.eos = 3
        self.mask = 4
        self.src_itos = {i:s for i,s in enumerate(src_vocab)}
        self.tgt_itos = {i:s for i,s in enumerate(tgt_vocab)}
        self.src_stoi = {s:i for i,s in enumerate(src_vocab)}
        self.tgt_stoi = {s:i for i,s in enumerate(tgt_vocab)}
        assert self.src_stoi['[pad]'] == self.tgt_stoi['[pad]'] == self.pad
        assert self.src_stoi['[unk]'] == self.tgt_stoi['[unk]'] == self.unk
        assert self.src_stoi['[bos]'] == self.tgt_stoi['[bos]'] == self.bos
        assert self.src_stoi['[eos]'] == self.tgt_stoi['[eos]'] == self.eos
        assert self.src_stoi['[mask]'] == self.tgt_stoi['[mask]'] == self.mask
        

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
    
    def tokenize_src(self, src):
        src = self._char_tokenize(src)
        src = self._token_to_id(src, self.src_stoi)
        return src

    def tokenize_tgt(self, tgt):
        tgt = self._word_tokenize(tgt)
        tgt = self._token_to_id(tgt, self.tgt_stoi)
        tgt = [self.bos] + tgt + [self.eos]
        return tgt

    def tokenize(self, src, tgt):
        src = self.tokenize_src(src)
        tgt = self.tokenize_tgt(tgt)
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


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


class Dataset(data.Dataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=50):
        self.tokenizer = tokenizer

        df = pd.read_csv(file_path, sep='\t', names=['src', 'tgt'])

        tokens = [tokenizer.tokenize(str(x.src), str(x.tgt)) for i, x in tqdm(df.iterrows())]
        self.src = [x[0] for x in tokens]
        self.tgt = [x[1] for x in tokens]

        self.src = pad_sequences(self.src, maxlen=src_pad_len, value=tokenizer.pad, padding='post')
        self.tgt = pad_sequences(self.tgt, maxlen=tgt_pad_len, value=tokenizer.pad, padding='post')


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


class NewMaskDataset(data.Dataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=50):
        self.tokenizer = tokenizer
        self.src_pad_len = src_pad_len
        self.tgt_pad_len = tgt_pad_len

        with open(file_path) as f:
            sent = f.read().split("\n")[:-1]

        tokens = [tokenizer.tokenize(str(x), str(x)) for x in tqdm(sent)]
        self.src = [x[0] for x in tokens]
        self.tgt = [x[1] for x in tokens]

        self.pad_value = self.tokenizer.pad
        self.mask_value = self.tokenizer.mask

        self.src_vocab_len = len(self.tokenizer.src_stoi)
        self.tgt_vocab_len = len(self.tokenizer.tgt_stoi)


    def mask_item(self, x,
        pad_value, mask_value, vocab_len,
        pc_mask = 0.2, pc_rep_mask = 0.8, pc_rep_other = 0.1, pc_rep_same = 0.1):
        mask = np.random.choice([0,1,2,3], len(x), 
            p=[1-pc_mask, pc_mask*pc_rep_mask, pc_mask*pc_rep_other, pc_mask*pc_rep_same])
        x = np.where(mask==1, mask_value, x)
        x = np.where(mask==2, np.random.randint(5, vocab_len), x)
        return x, mask

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = self.src[index]
        tgt_inp = self.tgt[index]
        tgt_lbl = self.tgt[index]

        src, src_mask = self.mask_item(src, self.pad_value, self.mask_value, self.src_vocab_len,
            pc_mask = 0.3, pc_rep_mask = 0.5, pc_rep_other = 0.5, pc_rep_same = 0.0)
        tgt_inp, tgt_mask = self.mask_item(tgt_inp, self.pad_value, self.mask_value, self.tgt_vocab_len,
            pc_mask = 0.3, pc_rep_mask = 0.7, pc_rep_other = 0.2, pc_rep_same = 0.1)

        pad_src = pad_sequences([src, src_mask], maxlen=self.src_pad_len, value=self.tokenizer.pad, padding='post')
        pad_tgt = pad_sequences([tgt_inp, tgt_lbl, tgt_mask], maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')

        src = pad_src[0]
        src_mask = pad_src[1]

        tgt_inp = pad_tgt[0]
        tgt_lbl = pad_tgt[1]
        tgt_mask = pad_tgt[2]

        return src, src_mask, tgt_inp, tgt_lbl, tgt_mask


class NewMaskDatasetFT(NewMaskDataset):
    def __init__(self, file_path, tokenizer, src_pad_len=200, tgt_pad_len=50):
        self.tokenizer = tokenizer

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

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = self.src[index]
        tgt_inp = self.tgt[index]
        tgt_lbl = self.tgt[index]

        src = np.where(src==self.tokenizer.unk, self.tokenizer.mask, src)

        src = pad_sequences([src], maxlen=self.src_pad_len, value=self.tokenizer.pad, padding='post')[0]
        tgt_inp = pad_sequences([tgt_inp], maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')[0]
        tgt_lbl = pad_sequences([tgt_lbl], maxlen=self.tgt_pad_len, value=self.tokenizer.pad, padding='post')[0]

        return src, tgt_inp, tgt_lbl
