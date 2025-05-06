import torch
from torch import nn


class LayerNorm(nn.Module):

    '''
        
    '''


    def __init__(self,d_model,eps=1e-12):
        super(LayerNorm,self).__init__()

        #define learnable parameter gamma and beta
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        #small constant to avoid division by zero
        self.eps = eps


    def forward(self,x):

        #mean over last dimension
        mean = x.mean(-1,keepdim=True)
        #variance over last dimension
        var = x.var(-1,unbiased=False,keepdim=True)

        #normalize
        out = (x-mean)/torch.sqrt(var+self.eps)
        #scale and shift
        out = self.gamma*out+self.beta


        return out