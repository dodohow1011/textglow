import sys
import torch
import torch.nn as nn
import numpy as np
from SubLayer import EncoderLayer

def position_encoding(n_position, d_hid, padding_idx=None):
    
    def cal_angle(position, hid_idx):
       return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, d_model, n_position, n_symbols, embedding_dim, n_head, d_hidden, d_k, d_v, n_layers, dropout=0.1):

        super(Encoder, self).__init__()

        self.d_model = d_model 
        self.n_position = n_position
        self.n_symbols = n_symbols # total num of word
        self.d_char_vec = embedding_dim # character embedding
        self.n_head = n_head
        self.d_inner = d_hidden 
        self.d_k = d_k # dimension of key
        self.d_v = d_v # dimension of value
        self.n_layers = n_layers
        self.dropout = dropout
        self.d_output = 256

        self.src_word_emb = nn.Embedding(self.n_symbols, self.d_char_vec, padding_idx=0)

        self.position_enc = nn.Embedding.from_pretrained(
            position_encoding(self.n_position, self.d_char_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, dropout=self.dropout)
            for _ in range(self.n_layers)])

        #self.linear = nn.Linear(self.d_char_vec, self.d_output)

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        max_length = src_seq.size(1)
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)[:, :max_length]

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        #enc_output = self.linear(enc_output)
        
        if return_attns:
            return enc_output, enc_slf_attn_list[-1]
        return enc_output
