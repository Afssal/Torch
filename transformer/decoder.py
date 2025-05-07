import torch
from torch import nn


from decoder_layer import DecoderLayer
from transformer_embedding import TransformerEmbedding



class Decoder(nn.Module):

    def __init__(self,dec_vocab_size,max_len,d_model,ffn_hidden,n_head,n_layers,drop_out,device):
        super().__init__()


        self.emb = TransformerEmbedding(d_model=d_model,drop_out=drop_out,max_len=max_len,
                                        vocab_size=dec_vocab_size,device=device)
        

        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,
                         n_head=n_head,drop_out=drop_out)
                         for _ in range(n_layers)
        ])

        self.linear = nn.Linear(d_model,dec_vocab_size)


    
    def forward(self,trg,enc_src,trg_mask,src_mask):

        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg,enc_src,trg_mask,src_mask)

        
        output = self.linear(trg)
        return output
    

    
