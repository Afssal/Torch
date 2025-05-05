import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


img_size = 128


composed = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])



train_dataset = datasets.ImageFolder(root='/home/afsal/Downloads/Torch/archive (2)/training_set', transform=composed)
test_dataset = datasets.ImageFolder(root='/home/afsal/Downloads/Torch/archive (2)/test_set',transform=composed)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_dataset,
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

        self.sigmoid = torch.nn.Sigmoid()



    def forward(self,x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        # x = self.sigmoid(x)

        return x
    


model = NeuralNetwork(3*128*128,1)
optimizer = torch.optim.SGD(
    model.parameters(),lr=0.01
)

criterion = torch.nn.BCEWithLogitsLoss()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for idx,(features,labels) in enumerate(train_loader):
        features = features.view(features.size(0),-1)
        logits = model(features)
        # Convert to float and reshape ex: [32] to [1,32]
        labels = labels.float().view(-1, 1)
        loss = criterion(logits,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} ----- Train loss : {loss}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.view(features.size(0), -1)
            labels = labels.float().view(-1, 1)

            outputs = model(features)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} - Test Accuracy: {acc:.2f}%")


#dataset loading 
"""
dataset structure

    data/
    ├── class1/
    │   ├── img1.png
    │   └── img2.png
    ├── class2/
    │   ├── img3.png
    │   └── img4.png

code
    from torch.utils.data import DataLoader, random_split
    
    dataset = datasets.ImageFolder(root='data', transform=transform)

    # Split into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
"""