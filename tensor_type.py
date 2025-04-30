import torch


tensor1d = torch.tensor([1.4,2.0,3.1])

#get tensor type
print(tensor1d.dtype)

#tensor type conversion
Ntensor = tensor1d.to(torch.int64)
print(Ntensor)


