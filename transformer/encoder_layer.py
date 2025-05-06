from torch import nn
from Layer_normalization import LayerNorm
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionwiseFeedback



class EncoderLayer(nn.Module):

    # EncoderLayer Architecture

    # Input: x (embedding), src_mask (mask for attention)

    # ┌────────────────────────────────────────────┐
    # │               Multi-Head Attention         │
    # │  - Input: x (as q, k, v), mask             │
    # │  - Output: attention output                │
    # └────────────────────────────────────────────┘
    #  └── Add & Norm:
    #      - Add residual connection (x + attention output)
    #      - Apply LayerNorm
    #      - Apply Dropout

    # ┌────────────────────────────────────────────┐
    # │        Position-wise Feed-Forward          │
    # │  - Two Linear layers with ReLU in between  │
    # └────────────────────────────────────────────┘
    #  └── Add & Norm:
    #      - Add residual connection (x + ffn output)
    #      - Apply LayerNorm
    #      - Apply Dropout

    # Output: x (same shape as input, but enriched with contextual info)



    def __init__(self,d_model,ffn_hidden,n_head,drop_out):
        super(EncoderLayer,self).__init__()

        self.attention = MultiHeadAttention(d_model=d_model,n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_out)


        self.ffn = PositionwiseFeedback(drop_out=drop_out,d_model=d_model,hidden=ffn_hidden)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_out)

    
    def forward(self,x,src_mask):

        _x = x
        x = self.attention(q=x,k=x,v=x,mask=src_mask)

        x = self.dropout1(x)
        x = self.norm1(x+_x)

        _x = x
        x = self.ffn(x)


        x = self.dropout2(x)
        x= self.norm2(x+_x)

        return x