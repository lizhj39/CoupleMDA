import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader


def datalodaer_gen(pos, neg, batch_size=128, shuffle=True):
    assert isinstance(pos, np.ndarray) and isinstance(neg, np.ndarray)
    # assert pos.shape == neg.shape

    x = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(pos.shape[0], dtype=np.int64),
                        np.zeros(neg.shape[0], dtype=np.int64)], axis=0)
    dataloader = DataLoader(TensorDataset(torch.from_numpy(x).long(), torch.from_numpy(y).float()),
                            batch_size=batch_size, shuffle=shuffle)
    return dataloader


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=-1e-6, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        parent_dir = os.path.dirname(save_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Performence increased ({self.val_loss_min:.6f} --> {abs(val_loss):.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = abs(val_loss)


import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # 创建一个位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 填充位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 增加一个batch维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]  # 只取前x.size(1)个位置编码


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_encoder_layers=2, max_len=5):
        super(TransformerEncoder, self).__init__()

        self.pos_encoder = nn.Embedding(max_len, d_model)

        # 定义Transformer Encoder层
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.shape[0], seq_len)  # (batch_size, seq_len)
        x = x + self.pos_encoder(position_ids)
        x = tfm_batch(self.transformer_encoder, x, small_bs=25000)
        return x         # (batch_size, seq_len, d_model)


def tfm_batch(transformer, x, small_bs=25000):
    batch_size, seq_len, d_model = x.shape
    outputs = []
    for i in range(0, batch_size, small_bs):
        small_batch = x[i:min(i + small_bs, batch_size), :, :]
        output = transformer(small_batch)
        outputs.append(output)
    return torch.cat(outputs, dim=0)


def mha_batch(multihead_attention, x, small_bs=25000):
    batch_size, seq_len, d_model = x.shape
    outputs = []
    for i in range(0, batch_size, small_bs):
        small_batch = x[i:min(i + small_bs, batch_size), :, :]
        output, _ = multihead_attention(small_batch, small_batch, small_batch, need_weights=False)
        outputs.append(output)
    return torch.cat(outputs, dim=0)


def mul_head_fc(fc, x: torch.Tensor):
    bs, num_heads, out_dim = x.shape
    return fc(x.view(-1, out_dim)).view(bs, num_heads, -1)


if __name__ == '__main__':
    model = TransformerEncoder(d_model=64, nhead=4, num_encoder_layers=2, max_len=5)
