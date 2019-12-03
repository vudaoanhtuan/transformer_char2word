import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_padding_mask(x, padding_value=0):
    # x: BxS
    mask = x==padding_value
    return mask

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                np.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * np.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len].clone().detach()
        if x.is_cuda:
            pe = pe.cuda()
        x = x + pe
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, src_vocab_len, tgt_vocab_len, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.src_padding_value = 0
        self.tgt_padding_value = 0

        self.pos_embedding = PositionalEncoder(d_model, dropout=dropout)
        self.src_embedding = nn.Embedding(src_vocab_len, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.linear_out = nn.Linear(d_model, tgt_vocab_len)

    def forward(self, src_inp, tgt_inp, tgt_lbl):
        # src_inp: BxS
        # tgt_inp: BxT
        # tgt_lbl: BxT

        # src padding mask: prevent attention weight from padding word
        src_padding_mask = generate_padding_mask(src_inp, self.src_padding_value) # BxS

        # tgt padding mask: prevent attention weight from padding word
        tgt_padding_mask = generate_padding_mask(tgt_inp, self.tgt_padding_value) # BxT

        # target self-attention mask, decoder side: word at position i depend only on word from 0 to i
        tgt_mask = generate_square_subsequent_mask(tgt_inp.shape[1]) # TxT

        # decoder-encoder mask for padding value in encoder side
        # memory_padding_mask = src_padding_mask

        if src_inp.is_cuda:
            src_padding_mask = src_padding_mask.cuda()
            tgt_padding_mask = tgt_padding_mask.cuda()
            tgt_mask = tgt_mask.cuda()

        src_inp = self.src_embedding(src_inp)
        src_inp = self.pos_embedding(src_inp) # BxSxE
        src_inp = src_inp.transpose(0,1) # SxBxE

        tgt_inp = self.tgt_embedding(tgt_inp)
        tgt_inp = self.pos_embedding(tgt_inp) # BxTxE
        tgt_inp = tgt_inp.transpose(0,1) # TxBxE

        import pdb; pdb.set_trace();
        memory = self.encoder(
            src_inp, 
            src_key_padding_mask=src_padding_mask
        )
        output = self.decoder(
            tgt_inp, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )


        output = output.transpose(0,1) # BxTxE
        output = self.linear_out(output) # BxTxV


        loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]), 
            tgt_lbl.reshape(-1), 
            ignore_index=self.tgt_padding_value
        )

        return output, loss

