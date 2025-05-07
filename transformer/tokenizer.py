import pandas as pd
from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer

df = pd.read_csv('/home/afsal/Downloads/Torch/data/translation_train.csv')
languages = ['english', 'german']

def data_corpus(lang):
    for i in range(len(df)):
        yield df[lang][i]

special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

for lang in languages:
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

    tokenizer.train_from_iterator(data_corpus(lang), trainer=trainer)
    tokenizer.save(f'{lang}.json')
new_tokenizer = Tokenizer.from_file("tokenizer.json")
