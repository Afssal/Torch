from torch import nn


from encoder_layer import EncoderLayer
from transformer_embedding import TransformerEmbedding



class Encoder(nn.Module):

    def __init__(self,enc_vocab_size,seq_len,d_model,ffn_hidden,n_head,n_layers,drop_out,device):
        super().__init__()

        self.emb = TransformerEmbedding(d_model=d_model,max_len=seq_len,
            vocab_size=enc_vocab_size,drop_out=drop_out,
            device=device
        )


        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,
                          n_head=n_head,drop_out=drop_out)
                          for _ in range(n_layers)]
        )


    def forward(self,x,src_mask):

        x = self.emb(x)

        for layer in self.layers:
            x = layer(x,src_mask)


        return x