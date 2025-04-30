import torch


#zero dimension
tensor0d = torch.tensor(1)

#single dimension
tensor1d = torch.tensor([1,2,3])

#two dimension
tensor2d = torch.tensor([[1,2],
                         [3,4]])

#three dimension
tensor3d = torch.tensor([
    [
        [1,2],[3,4]
    ],
    [
        [5,6],[7,8]
    ]
])

print(tensor0d)
print(tensor1d)
print(tensor2d)
print(tensor3d)