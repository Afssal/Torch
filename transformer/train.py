import math
import time
import torch
from torch import nn,optim
from torch.optim import Adam


from csv_loader import CsvDataloader,vocabulary
from transformer1 import Transformer
from evaluate import idx_to_word,get_bleu
from epoch_timer import epoch_time



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):

    if hasattr(m,'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

src,tgt = vocabulary()

model = Transformer(
    src_pad_idx=1,
    trg_pad_idx=1,
    trg_sos_idx=2,
    enc_vocab_size=14520,
    dec_vocab_size=25000,
    max_len=256,
    ffn_hidden=2048,
    n_head=8,
    n_layers=6,
    drop_out=0.1,
    d_model=512,
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
)

print(f"model has {count_parameters(model):,} trainable parameters")
model.apply(initialize_weights)
optimizers = Adam(params=model.parameters(),
                  lr=1e-5,
                  weight_decay=5e-4,
                  eps=5e-9)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizers,
    factor=0.9,
    patience=10
)

criterion = nn.CrossEntropyLoss(ignore_index=1)

def train(model,iterator,optimizer,criterion,clip):

    model.train()
    epoch_loss = 0
    for i,(src,trg) in enumerate(iterator):

        # src = batch.src
        # trg = batch.trg

        optimizer.zero_grad()
        output = model(src,trg[:,:-1])

        output_reshape = output.contiguous().view(-1,output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)

        loss = criterion(output_reshape,trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()


        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())


    return epoch_loss / len(iterator)


def evaluate(model,iterator,criterion):

    model.eval()
    epoch_loss = 0
    batch_bleu = []
    batch_size = 128
    with torch.no_grad():
        for i,(src,trg) in enumerate(iterator):
            # src = batch.src
            # trg = batch.trg
            output = model(src,trg[:,:-1])
            output_shape = output.contiguous().view(-1,output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)

            loss = criterion(output_shape,trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):

                try:
                    trg_words = idx_to_word(trg[j],src.get_vocab())
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words,tgt.get_vocab())
                    bleu = get_bleu(hypotheses=output_words.split(),reference=trg_words.split())
                    total_bleu.append(bleu)

                except:
                    pass

                total_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(total_bleu)
    return epoch_loss / len(iterator),batch_bleu


def run(total_epoch,best_loss):

    train_losses,test_losses,bleus = [],[],[]
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model,CsvDataloader(),optimizers,criterion,clip=1.0)
        valid_loss,bleu = evaluate(model,CsvDataloader(),criterion)
        end_time = time.time()


        if step > 100:

            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins,epoch_secs = epoch_time(start_time,end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(),'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')



run(total_epoch=5, best_loss=float('inf'))