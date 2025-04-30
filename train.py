import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

torch.manual_seed(123)

x_train = torch.tensor(
    [
        [-1.2,3.1],
        [-0.9,2.9],
        [-0.5,2.6],
        [2.3,-1.1],
        [2.7,-1.5]
    ]
)

y_train = torch.tensor([0,0,0,1,1])

x_test = torch.tensor(
    [
        [-0.8,2.8],
        [2.6,-1.6]
    ]
)

y_test = torch.tensor([0,1])


#define dataset class how to load data to model

class ToyDataset(Dataset):

    def __init__(self,x,y):
        self.features = x
        self.labels = y

    def __getitem__(self,index):
        one_x = self.features[index]
        one_y = self.features[index]
        return one_x,one_y
    
    def __len__(self):
        return self.labels.shape[0]
    

train_ds = ToyDataset(x_train,y_train)
test_ds = ToyDataset(x_test,y_test)


#define dataloader that handle data shuffling, loading data into batches
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True #drop last batch if batch element not equal to other batch
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    drop_last=True
)


# for idx,(x,y) in enumerate(train_loader):
#     print(idx,x,y)


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
    

model = NeuralNetwork(2,2)
optimizer = torch.optim.SGD(
    model.parameters(),lr=0.02
)
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for idx,(features,labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch : {epoch} ---- Train loss : {loss}")

    
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
    print(outputs)

#save model
torch.save(model.state_dict(),"model.pth")

#load model
model.load_state_dict(torch.load("model.pth"))
