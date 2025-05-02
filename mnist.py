import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


img_size=16

composed = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])

train_data = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=composed)
test_data = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=composed)


train_loader = DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    drop_last=True
)


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
    
model = NeuralNetwork(16*16,10)
optimizer = torch.optim.SGD(
    model.parameters(),lr=0.01
)
criterion = torch.nn.CrossEntropyLoss()



num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for idx,(features,labels) in enumerate(train_loader):
        # change shape [batch_size, 1, 16, 16] to [batch_size, 256]
        features = features.view(features.size(0), -1)
        logits = model(features)
        loss = criterion(logits,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch : {epoch} ---- Train loss : {loss}")


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feat,lab in test_loader:
            feat = feat.view(feat.size(0),-1)
            outputs = model(feat)
            _,predicted = torch.max(outputs.data,1)
            total += lab.size(0)
            print("lab",lab.size(0))
            correct += (predicted == lab).sum().item()

    print(f"Epoch {epoch} - Test Accuracy: {100 * correct / total:.2f}%")