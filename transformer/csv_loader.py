import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence


class Translation(Dataset):

    def __init__(self,):
        self.df = pd.read_csv('/home/afsal/Downloads/Torch/data/translation_train.csv')
        self.src = self.df['english']
        self.tgt = self.df['german']
        self.src_token = Tokenizer.from_file('/home/afsal/Downloads/Torch/transformer/english.json')
        self.tgt_token = Tokenizer.from_file('/home/afsal/Downloads/Torch/transformer/german.json')


    def __getitem__(self,index):

        source = self.src.iloc[index].strip()
        target = self.tgt.iloc[index].strip()
        src_id = self.src_token.encode(source).ids
        tgt_id = self.tgt_token.encode(target).ids

        return torch.tensor(src_id),torch.tensor(tgt_id)
    
    def __len__(self):
        return len(self.df)
# print(src_token.token_to_id("[PAD]"))


def collate_fn(batch):

    src_batch,tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch,padding_value=1,batch_first=True)
    tgt_batch = pad_sequence(tgt_batch,padding_value=1,batch_first=True)

    return src_batch,tgt_batch

dataset = Translation()
dataloader = DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)

for src,tgt in dataloader:
    print(src,tgt)

# for src,tgt in translates:
#     print(src,tgt)



