import torch
import pandas as pd
from torch.utils.data import Dataset


class Translation(Dataset):

    def __init__(self,):
        self.df = pd.read_csv('/home/afsal/Downloads/Torch/data/translation_train.csv')
        self.src = self.df['english']
        self.tgt = self.df['german']


    def __getitem__(self,index):

        source = self.src.iloc[index].strip()
        target = self.tgt.iloc[index].strip()

        return source,target

translates = Translation()

for src,tgt in translates:
    print(src,tgt)



