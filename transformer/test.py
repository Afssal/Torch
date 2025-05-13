
from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer


tokenizer = Tokenizer.from_file('/home/afsal/Downloads/Torch/transformer/german.json')


pad_id =  tokenizer.token_to_id("[PAD]")
print(pad_id)