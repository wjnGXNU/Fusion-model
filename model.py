import numpy as np
import torch
import torch.nn as nn

from block import get_sinusoid_encoding_table, EncoderLayer, get_non_pad_mask, get_attn_key_pad_mask
from config import SIG_LEN, d_model, d_inner, num_layers, num_heads, KS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#vit模型
class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(
            self,
            d_feature,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()
        n_position = d_feature + 1
        self.src_word_emb = nn.Conv1d(1, d_model, kernel_size=KS, padding=int((KS - 1) / 2))
        #position_encoding
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
    def forward(self, src_seq):
        np.set_printoptions(threshold=np.inf)
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        np.set_printoptions(threshold=np.inf)
        enc_output = src_seq.unsqueeze(1)
        np.set_printoptions(threshold=np.inf)
        enc_output = self.src_word_emb(enc_output)
        enc_output = enc_output.transpose(1, 2)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output,
class Transformer(nn.Module):
    def __init__(
            self, device,
            d_feature, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            class_num=5):
        super().__init__()
        self.encoder = Encoder(d_feature, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.device = device
        self.linear1_cov = nn.Conv1d(d_feature, 1, kernel_size=1)
        self.linear1_linear = nn.Linear(68, 64)
        self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
        self.linear2_linear = nn.Linear(d_feature, class_num)
    def forward(self, src_seq, RR):
        enc_output, *_ = self.encoder(src_seq)
        dec_output = enc_output
        res = self.linear1_cov(dec_output)
        res = res.contiguous().view(res.size()[0], -1)
        return res
