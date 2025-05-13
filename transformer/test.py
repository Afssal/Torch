
from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer
from csv_loader import vocabulary,CsvDataloader

tokenizer = Tokenizer.from_file('/home/afsal/Downloads/Torch/transformer/english.json')

src,tgt = vocabulary()

pad_id =  tokenizer.token_to_id("[SOS]")
print(src.get_vocab())

for src,tgt in CsvDataloader():
    print(src,tgt)