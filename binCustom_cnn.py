import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets



img_size = 128


composed = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])


train_dataset = datasets.ImageFolder(root='/home/afsal/Downloads/Torch/archive (2)/training_set',transform=composed)
valid_dataset = datasets.ImageFolder(root='/home/afsal/Downloads/Torch/archive (2)/test_set',transform=composed)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    drop_last=True
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    drop_last=True
)


class CNN(torch.nn.Module):


    def __init__(self,out1,out2):
        super().__init__()


        self.cnn1 = torch.nn.Conv2d(in_channels=3,out_channels=out1,kernel_size=5,stride=1,padding=2)
        self.max = torch.nn.MaxPool2d(kernel_size=2)
        self.cnn2 = torch.nn.Conv2d(in_channels=out1,out_channels=out2,kernel_size=5,stride=1,padding=2)
        self.fc1 = nn.Linear(out2*(img_size // 4) * (img_size // 4),1)

    
    def forward(self,x):
        x = self.cnn1(x)
        x = self.max(x)
        x = self.cnn2(x)
        x = self.max(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)

        return x
    

model = CNN(out1=16,out2=32)
criterion = torch.nn.BCEWithLogitsLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

train_loader = DataLoader(dataset=train_dataset,batch_size=32)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=32)


n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    for idx,(features,labels) in enumerate(train_loader):
        labels = labels.float().unsqueeze(1)
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
        for features,labels in valid_loader:
            outputs = model(features)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)


    acc = 100*correct/total
    print(f"Epoch {epoch} - Test Accuracy : {acc:2f}%")    