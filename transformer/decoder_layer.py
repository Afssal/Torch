from torch import nn

from Layer_normalization import LayerNorm
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionwiseFeedback




class DecoderLayer(nn.Module):


    # DecoderLayer Architecture

    # Input: 
    # - dec: decoder input (target embeddings)
    # - enc: encoder output
    # - trg_mask: mask for decoder self-attention (usually for future tokens)
    # - src_mask: mask for encoder-decoder attention

    # ┌────────────────────────────────────────────┐
    # │         Masked Multi-Head Attention        │
    # │  - Input: dec (as q, k, v), trg_mask       │
    # │  - Output: self-attention output           │
    # └────────────────────────────────────────────┘
    #  └── Add & Norm:
    #      - Add residual connection (dec + self-attn)
    #      - Apply LayerNorm
    #      - Apply Dropout

    # ┌────────────────────────────────────────────┐
    # │       Encoder-Decoder Multi-Head Attn      │
    # │  - Input: q = decoder output               │
    # │           k, v = encoder output            │
    # │  - Mask: src_mask                          │
    # └────────────────────────────────────────────┘
    #  └── Add & Norm:
    #      - Add residual connection (x + enc-dec attn)
    #      - Apply LayerNorm
    #      - Apply Dropout

    # ┌────────────────────────────────────────────┐
    # │        Position-wise Feed-Forward          │
    # │  - Two Linear layers with ReLU in between  │
    # └────────────────────────────────────────────┘
    #  └── Add & Norm:
    #      - Add residual connection (x + ffn)
    #      - Apply LayerNorm
    #      - Apply Dropout

    # Output: x (same shape as input, enriched with encoder and target context)



    def __init__(self,d_model,ffn_hidden,n_head,drop_out):
        super(DecoderLayer,self).__init__()


        self.self_attention = MultiHeadAttention(d_model=d_model,n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_out)


        self.enc_dec_attention = MultiHeadAttention(d_model=d_model,n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_out)



        self.ffn = PositionwiseFeedback(d_model=d_model,hidden=ffn_hidden,drop_out=drop_out)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_out)


    
    def forward(self,dec,enc,trg_mask,src_mask):


        _x = dec
        x = self.self_attention(q=dec,k=dec,v=dec,mask=trg_mask)

        x = self.dropout1(x)
        x = self.norm1(x+_x)

        if enc is not None:

            _x = x
            x = self.enc_dec_attention(q=x,k=enc,v=enc,mask=src_mask)

            x = self.dropout2(x)

            x = self.norm2(x+_x)


        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x+_x)

        return x