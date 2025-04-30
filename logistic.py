import torch.nn.functional as F
import torch


torch.cuda.is_available()

y = torch.tensor([1.0])
x = torch.tensor([1.1])
#requires_grad=True; telling we need to calculate gradient of given variables
w = torch.tensor([2.2],requires_grad=True)
b = torch.tensor([0.0],requires_grad=True)

#forward pass

#output calculation
z = x*w+b
a = torch.sigmoid(z)


#loss calculation
loss = F.binary_cross_entropy(a,y)

print(loss)


#backpropagation
loss.backward()

print(w.grad)
print(b.grad)

