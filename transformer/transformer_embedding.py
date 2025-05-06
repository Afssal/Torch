from torch import nn

from positional_encoding import PositionalEncoding
from token_embedding import TokenEmbedding



class TransformerEmbedding(nn.Module):


    '''
    the input to encoder and decoder is calculated here
    based on transformer architecture, we sum output from input embedding 
    with positional encoding 
    input shapes
    token embedding : (vocabulary_size,dimension_size)
    position encoding : (dimension_size,sequence_length)
    
    '''


    def __init__(self,vocab_size,d_model,max_len,drop_out,device):
        super(TransformerEmbedding,self).__init__()

        self.tok_emb = TokenEmbedding(vocab_size,d_model)
        self.pos_emb = PositionalEncoding(d_model,max_len,device)
        self.dropout = nn.Dropout(p=drop_out)


    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)

        return self.dropout(tok_emb+pos_emb)