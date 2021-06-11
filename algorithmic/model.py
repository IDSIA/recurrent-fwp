# Contains model implementations.

import torch
import torch.nn as nn

from layers import PositionalEncoding, TransformerFFlayers
from layers import (
    FastFFlayer, FastRNNlayer, FastFFslowRNNlayer, RecUpdateTanhFastFFlayer,
    LinearAttentionlayer, Attentionlayer)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        # More efficient than optimizer.zero_grad() according to:
        # Szymon Migacz "PYTORCH PERFORMANCE TUNING GUIDE" at GTC-21.
        # - doesn't execute memset for every parameter
        # - memory is zeroed-out by the allocator in a more efficient way
        # - backward pass updates gradients with "=" operator (write) (unlike
        # zero_grad() which would result in "+=").
        # In PyT >= 1.7, one can do `model.zero_grad(set_to_none=True)`
        for p in self.parameters():
            p.grad = None

    def print_params(self):
        for p in self.named_parameters():
            print(p)


# Pure PyTorch LSTM model
class LSTMModel(BaseModel):
    def __init__(self, emb_dim, hidden_size, in_vocab_size, out_vocab_size,
                 dropout=0.0, num_layers=1):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=in_vocab_size, embedding_dim=emb_dim)

        self.rnn_func = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size,
                                num_layers=num_layers)

        self.dropout = dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        out = self.embedding(x).permute(1, 0, 2)  # seq dim first

        if self.dropout:
            out = self.dropout(out)
        out, _ = self.rnn_func(out)

        if self.dropout:
            out = self.dropout(out)
        logits = self.out_layer(out).permute(1, 0, 2)

        return logits


# Transformer model based on PyTorch `TransformerEncoder`.
# contains a redundant layernorm in the final layer.
class TrafoModel(BaseModel):
    def __init__(self, hidden_size, in_vocab_size, out_vocab_size, dropout=0.0,
                 nheads=8, num_layers=6, ff_factor=4, use_pos_enc=True):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(hidden_size)

        # embedding_dim = hidden_size for residual connection
        self.embedding = nn.Embedding(
            num_embeddings=in_vocab_size, embedding_dim=hidden_size)

        ff_dim = hidden_size * ff_factor

        t_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nheads, dim_feedforward=ff_dim,
            dropout=dropout, activation="relu")

        lnorm = nn.LayerNorm(hidden_size)

        self.transformer = nn.TransformerEncoder(t_layer, num_layers, lnorm)

        self.linear = nn.Linear(hidden_size, out_vocab_size)

        # Using default init:
        # normal for embedding
        # kaiming uniform for all other linear
        # 1 scale 0 bias for layer norm

    def get_mask(self, input, device='cuda'):
        # Generate an auto-regressive mask
        # Be careful with torch versions.
        slen = input.shape[1]

        # Copied from nn.Transformer generate_square_subsequent_mask (1.8.1):
        mask = (torch.triu(
            torch.ones(slen, slen, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(
            mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def forward(self, x):
        out = self.embedding(x).permute(1, 0, 2)  # seq dim first
        if self.use_pos_enc:
            out = self.pos_enc(out)

        mask = self.get_mask(x, device=x.device)
        out = self.transformer(src=out, mask=mask, src_key_padding_mask=None)

        logits = self.linear(out).permute(1, 0, 2)

        return logits


# Own Transformer implementation.
class OwnTransformer(BaseModel):
    def __init__(self, in_vocab_size, out_vocab_size, hidden_size,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, use_pos_enc=False):
        super(OwnTransformer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=in_vocab_size, embedding_dim=hidden_size)

        layers = []

        if use_pos_enc:
            layers.append(PositionalEncoding(hidden_size))

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                Attentionlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        out = self.embedding(x).permute(1, 0, 2)  # seq dim first
        out = self.layers(out)
        out = self.out_layer(out).permute(1, 0, 2)
        return out


# Linear Transformer with sum update rule.
class LinearTransformer(BaseModel):
    def __init__(self, in_vocab_size, out_vocab_size, hidden_size,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, use_pos_enc=False):
        super(LinearTransformer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=in_vocab_size, embedding_dim=hidden_size)

        layers = []

        if use_pos_enc:
            layers.append(PositionalEncoding(hidden_size))

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                LinearAttentionlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        out = self.embedding(x).permute(1, 0, 2)  # seq dim first
        out = self.layers(out)
        out = self.out_layer(out).permute(1, 0, 2)
        return out


# Linear Transformer with the delta update rule.
class DeltaNetModel(BaseModel):
    def __init__(self, in_vocab_size, out_vocab_size, hidden_size,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, use_pos_enc=False):
        super(DeltaNetModel, self).__init__()
        assert num_head * dim_head == hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=in_vocab_size, embedding_dim=hidden_size)

        layers = []

        if use_pos_enc:
            layers.append(PositionalEncoding(hidden_size))

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                FastFFlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        out = self.embedding(x).permute(1, 0, 2)  # seq dim first
        out = self.layers(out)
        out = self.out_layer(out).permute(1, 0, 2)
        return out


# Linear Transformer with Fast weight memory update rule.
class FastFFslowRNNModel(BaseModel):
    def __init__(self, in_vocab_size, out_vocab_size, hidden_size,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, use_pos_enc=False):
        super(FastFFslowRNNModel, self).__init__()
        assert num_head * dim_head == hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=in_vocab_size, embedding_dim=hidden_size)

        layers = []

        if use_pos_enc:
            layers.append(PositionalEncoding(hidden_size))

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                FastFFslowRNNlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        out = self.embedding(x).permute(1, 0, 2)  # seq dim first
        out = self.layers(out)
        out = self.out_layer(out).permute(1, 0, 2)
        return out


class FastRNNModel(BaseModel):
    def __init__(self, in_vocab_size, out_vocab_size, hidden_size,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, use_pos_enc=False):
        super(FastRNNModel, self).__init__()
        assert num_head * dim_head == hidden_size
        print(f"num_layers: {num_layers}")
        print(f"num_head: {num_head}")
        print(f"dim_head: {dim_head}")
        print(f"dim_ff: {dim_ff}")
        print(f"dropout: {dropout}")

        self.embedding = nn.Embedding(
            num_embeddings=in_vocab_size, embedding_dim=hidden_size)

        layers = []

        if use_pos_enc:
            layers.append(PositionalEncoding(hidden_size))

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                FastRNNlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        out = self.embedding(x).permute(1, 0, 2)  # seq dim first
        out = self.layers(out)
        out = self.out_layer(out).permute(1, 0, 2)
        return out


# Linear Transformer with Fast weight memory update rule.
class RecDeltaNetModel(BaseModel):
    def __init__(self, in_vocab_size, out_vocab_size, hidden_size,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, use_pos_enc=False):
        super(RecDeltaNetModel, self).__init__()
        assert num_head * dim_head == hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=in_vocab_size, embedding_dim=hidden_size)

        layers = []

        if use_pos_enc:
            layers.append(PositionalEncoding(hidden_size))

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                RecUpdateTanhFastFFlayer(
                    num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        out = self.embedding(x).permute(1, 0, 2)  # seq dim first
        out = self.layers(out)
        out = self.out_layer(out).permute(1, 0, 2)
        return out


if __name__ == '__main__':
    from datetime import datetime

    torch.manual_seed(111)

    in_vocab_size = 10
    out_vocab_size = 12

    emb_dim = 11
    hidden_size = 16
    dropout = 0.1
    nheads = 2
    dim_head = 8
    num_layers = 2
    ff_factor = 2
    use_pos_enc = False

    random_input = torch.tensor(
        [[6, 8, 1, 0, 4, 0, 1],
         [2, 6, 5, 7, 3, 8, 7],
         [7, 6, 3, 5, 5, 8, 1],
         [8, 2, 4, 5, 0, 8, 7],
         [8, 1, 5, 7, 0, 6, 4],
         [8, 6, 3, 6, 5, 6, 7],
         [8, 5, 3, 7, 6, 0, 2],
         [5, 2, 4, 1, 5, 7, 7],
         [3, 5, 2, 6, 4, 4, 5],
         [1, 3, 3, 5, 0, 3, 2],
         [3, 1, 1, 1, 6, 1, 6],
         [7, 6, 5, 7, 0, 6, 5],
         [3, 4, 8, 7, 8, 3, 7],
         [5, 8, 5, 3, 5, 3, 0],
         [1, 4, 4, 1, 3, 5, 7]], device='cuda')

    print("========================")
    print(f"  Test LSTMModel  {datetime.now()}")
    print("========================")
    model = LSTMModel(emb_dim=emb_dim, hidden_size=hidden_size,
                      num_layers=num_layers, in_vocab_size=in_vocab_size,
                      out_vocab_size=out_vocab_size, dropout=dropout)

    print("Random parameters")
    model.print_params()

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))

    print("========================")
    print(f"  Test TrafoModel  {datetime.now()}")
    print("========================")

    model = TrafoModel(hidden_size=hidden_size, in_vocab_size=in_vocab_size,
                       out_vocab_size=out_vocab_size, dropout=dropout,
                       nheads=nheads, num_layers=num_layers,
                       ff_factor=ff_factor)

    print("Random parameters")
    model.print_params()

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))

    print("========================")
    print(f"  Test LinearTransformer  {datetime.now()}")
    print("========================")

    model = LinearTransformer(in_vocab_size=in_vocab_size,
                              out_vocab_size=out_vocab_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              num_head=nheads, dim_head=dim_head,
                              dim_ff=ff_factor * hidden_size,
                              dropout=dropout, use_pos_enc=use_pos_enc)

    print("Random parameters")
    model.print_params()

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))

    print("========================")
    print(f"  Test FastRNNModel  {datetime.now()}")
    print("========================")

    model = FastRNNModel(in_vocab_size=in_vocab_size,
                         out_vocab_size=out_vocab_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         num_head=nheads, dim_head=dim_head,
                         dim_ff=ff_factor * hidden_size,
                         dropout=dropout, use_pos_enc=use_pos_enc)

    print("Random parameters")
    model.print_params()

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))

    print("========================")
    print(f"  Test RecDeltaNetModel  {datetime.now()}")
    print("========================")

    model = RecDeltaNetModel(in_vocab_size=in_vocab_size,
                                out_vocab_size=out_vocab_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=nheads, dim_head=dim_head,
                                dim_ff=ff_factor * hidden_size,
                                dropout=dropout, use_pos_enc=use_pos_enc)

    print("Random parameters")
    model.print_params()

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))
