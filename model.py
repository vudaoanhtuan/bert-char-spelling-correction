import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CombineLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, x, y):
        return self.a*x + self.b*y

class Model(nn.Module):
    def __init__(self, char_vocab_len, word_vocab_len,
                d_model=256, nhead=8,
                num_encoder_layers=3,
                dim_feedforward=1024,
                dropout=0.1, activation="relu",
                init_param=True):
        super().__init__()
        self.padding_value = 0

        self.pos_embedding = PositionalEncoder(d_model, dropout=dropout)
        self.word_embedding = nn.Embedding(word_vocab_len, d_model)

        self.char_linear = nn.Linear(char_vocab_len, d_model)
        self.combine_layer = CombineLayer()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.linear_out = nn.Linear(d_model, word_vocab_len)

        if init_param:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


    def encode(self, bert_input, bert_input_char, padding_mask=None):
        # bert_input: BxS
        # bert_input_char: BxSxchar_vocab
        # padding_mask: BxS
        # E = d_model

        bert_input = self.word_embedding(bert_input)
        bert_input = self.pos_embedding(bert_input) # BxSxE
        bert_input_char = self.char_linear(bert_input_char) # BxSxE
        combined_input = self.combine_layer(bert_input, bert_input_char) # BxSxE
        combined_input = combined_input.transpose(0,1) # SxBxE

        memory = self.encoder(
            combined_input, 
            src_key_padding_mask=padding_mask
        ) # SxBxd_model

        memory = memory.transpose(0,1) # BxSxd_model

        return memory # BxSxd_model

    def forward(self, bert_input, bert_input_char, bert_label=None):
        # bert_input: BxS
        # bert_input_char: BxSxchar_vocab
        # bert_label: BxS

        # src padding mask: prevent attention weight from padding word
        padding_mask = generate_padding_mask(bert_input, self.padding_value) # BxS

        if bert_input.is_cuda:
            padding_mask = src_padding_mask.cuda()

        encoder_output = self.encode(bert_input, bert_input_char, padding_mask=padding_mask) # BxSxd_model
        logit = self.linear_out(encoder_output)

        loss = None
        if bert_label is not None:
            loss = F.cross_entropy(
                logit.reshape(-1, logit.shape[-1]), 
                bert_label.reshape(-1), 
                ignore_index=self.padding_value
            )

        return logit, loss

