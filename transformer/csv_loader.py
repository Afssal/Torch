import torch
import pandas as pd
from torch.utils.data import Dataset
from tokenizers import Tokenizer


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

        return src_id,tgt_id

translates = Translation()

for src,tgt in translates:
    print(src,tgt)



