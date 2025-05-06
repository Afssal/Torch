import math
import torch
from torch import nn



class ScaledDotProductAttention(nn.Module):


    '''
        calculating attention score 
        k = key : q = query : v = value
        shape of k,q,v are : (batch size,head,sequence length,embedding dimension)
        attention score = softmax((QK^T)/sqrt(d_k))v
    '''
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()
        

        self.softmax = nn.Softmax(dim=-1)


    def forward(self,q,k,v,mask=None,e=1e-12):

        batch_size,head,length,d_tensor = k.size()

        #swapping dimenstions 2 and 3 (batch,head,d_tensor,length)
        k_t = k.transpose(2,3)

        #scaled dot product
        score = (q @ k_t)/math.sqrt(d_tensor)


        if mask is not None:
            score = score.masked_fill(mask == 0,-10000)

        #calculate softmax
        score = self.softmax(score)

        #multiply with value
        v = score @ v


        return v,score
    

batch_size = 2
num_heads = 1
seq_length = 4
d_tensor = 8


q = torch.randn(batch_size,num_heads,seq_length,d_tensor)
k = torch.randn(batch_size,num_heads,seq_length,d_tensor)
v = torch.randn(batch_size,num_heads,seq_length,d_tensor)



attention = ScaledDotProductAttention()


output, atten_score = attention(q,k,v)


print(f"Output shape : {output.shape}")
print(f"attention score shape : {atten_score.shape}")


print(atten_score)
