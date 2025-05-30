import torch
from torch import nn

from decoder import Decoder
from encoder import Encoder



class Transformer(nn.Module):


    def __init__(self,src_pad_idx,trg_pad_idx,trg_sos_idx,enc_vocab_size,dec_vocab_size,d_model,
                 n_head,max_len,ffn_hidden,n_layers,drop_out,device):
        super().__init__()



        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,n_head=n_head,seq_len=max_len,
                               ffn_hidden=ffn_hidden,enc_vocab_size=enc_vocab_size,drop_out=drop_out,
                               n_layers=n_layers,device=device)
        

        self.decoder = Decoder(d_model=d_model,n_head=n_head,max_len=max_len,ffn_hidden=ffn_hidden,
                               dec_vocab_size=dec_vocab_size,drop_out=drop_out,n_layers=n_layers,
                               device=device)
        

    
    def forward(self,src,trg):

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src,src_mask)
        output = self.decoder(trg,enc_src,trg_mask,src_mask)
        return output
    
    #function used to create a mask to hide padding token during attention
    def make_src_mask(self,src):

        #unsqueeze is done to reshape to [batch_size,1,1,src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    #two uses : use to ignore padded token and block future word so that at time t, decoder can see only word up to t, not after
    def make_trg_mask(self,trg):

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        #tril function used fill above diagonal element with 0
        trg_sub_mask = torch.tril(torch.ones(trg_len,trg_len)).type(torch.ByteTensor).to(self.device)
        #combine original matrix with tril matrix
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask