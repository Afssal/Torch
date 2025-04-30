import torch


tensor2d = torch.tensor([[1,2,3],
                         [4,5,6]])


#get tensor shape
print(tensor2d.shape)


#reshape tensor
print(tensor2d.reshape(3,2))

print(tensor2d.view(3,2))


#reshape = make copy and reshape the copy
#view = make change on original data


#transpose
print(tensor2d.T)


#multiplication
print(tensor2d.matmul(tensor2d.T))

print(tensor2d @ tensor2d.T)