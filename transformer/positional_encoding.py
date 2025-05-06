import torch
from torch import nn




class PositionalEncoding(nn.Module):

    '''
        d_model - dimension of the embeddings.
        max_len - maximum possible sequence length.
        device: the device (CPU/GPU) on which tensors will be allocated
    '''


    def __init__(self,d_model,max_len,device):

        super(PositionalEncoding,self).__init__()

        #create position encoding matrix
        
        #create tensor of shape [max_len,d_model] filled with zero
        self.encoding = torch.zeros(max_len,d_model,device=device)
        #no need to train
        self.encoding.requires_grad = False

        #generate positions and dimensions\
        #column vector shape [max_len,1] : column represent embedding position (upto 512)
        pos = torch.arange(0,max_len,device=device)
        pos = pos.float().unsqueeze(dim=1)

        #row vector : row represent sequence position (position of each word)
        _2i = torch.arange(0,d_model,step=2,device=device).float()


        #compute positional encoding
        #for even : sin(position/(10000^(2i/d_model)))
        self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        #for odd : cos(position/(10000^(2i/d_modl)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))


    def forward(self,x):

        #get batch size and sequence length
        batch_size,seq_len = x.size()

        #return only first seq_len row
        return self.encoding[:seq_len,:]
    
device = torch.device('cpu')
d_model = 8
max_len = 20


pos_enc = PositionalEncoding(d_model,max_len,device)

x = torch.zeros((1,10),device=device)

print(f"Input data {x}")


encoding = pos_enc(x)


print(f"Encoding Shape : {encoding.shape}")
print(f"Encoding output : {encoding}")