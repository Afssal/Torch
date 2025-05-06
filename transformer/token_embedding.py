from torch import nn



class TokenEmbedding(nn.Embedding):


    '''
        vocab_size : vocabulary size
        d_model : embedding dimenstion
        Also we are inherting the property of embedding layer here
    
    '''

    def __init__(self,vocab_size,d_model):
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)