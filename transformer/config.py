import torch


max_len = 256
ffn_hidden = 2048
n_head = 8
n_layers = 6
drop_out = 0.1
d_model = 512
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
lr = 1e-5
weight_decay = 5e-4
eps = 5e-9
factor = 0.9
patience = 10
batch_size =128
