import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader



img_size = 16

composed = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])

train_dataset=dsets.MNIST(root='./data',train=True,download=True,transform=composed)
validation_dataset=dsets.MNIST(root='./data',train=False,download=True,transform=composed)


class CNN(nn.Module):

    def __init__(self,out1,out2):
        super(CNN,self).__init__()


        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=out1,kernel_size=5,stride=1,padding=2)
        self.max = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=out1,out_channels=out2,kernel_size=5,stride=1,padding=2)
        self.fc1 = nn.Linear(out2*4*4,10)

    
    def forward(self,x):
        x = self.cnn1(x)
        x = self.max(x)
        x = self.cnn2(x)
        x = self.max(x)
        x= x.view(x.size(0),-1)
        x = self.fc1(x)
        return x
    


model = CNN(out1=16,out2=32)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

train_loader = DataLoader(dataset=train_dataset,batch_size=32)
valid_loader = DataLoader(dataset=validation_dataset,batch_size=32)


n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    for idx,(features,labels) in enumerate(train_loader):
        logits = model(features)
        loss = criterion(logits,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(f"Epoch : {epoch} ----- Train Loss : {loss}")


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feat,lab in valid_loader:
            outputs = model(feat)
            _,predicted = torch.max(outputs.data,1)
            total += lab.size(0)
            correct += (predicted == lab).sum().item()


    print(f"Epoch {epoch} - Test Accuracy : {100*correct/total:.2f}%")



     
