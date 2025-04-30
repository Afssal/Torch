import torch

class NeuralNetwork(torch.nn.Module):

    def __init__(self,num_inputs,num_outputs):
        super().__init__()


        self.layer1 = torch.nn.Linear(num_inputs,30)
        self.relu1 = torch.nn.ReLU()

        self.layer2 = torch.nn.Linear(30,20)
        self.relu2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Linear(20,num_outputs)


    def forward(self,x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x
    

model = NeuralNetwork(50,3)

#get model summary
print(model)

#get number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)

#get layer1 weights
print(model.layer1.weight)
print(model.layer1.weight.shape)

torch.manual_seed(123)
x = torch.rand((1,50))
out = model(x)
print(out)

#pytorch inference
with torch.no_grad():
    out = model(x)

print(out)