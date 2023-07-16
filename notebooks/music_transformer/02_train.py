# -*- coding: utf-8 -*-
# Data Preprocess

# Setup environment

import os
import pickle
import random
import secrets
import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn import metrics

torch.set_float32_matmul_precision('high')

import TMIDIX

from x_transformer import TransformerWrapper, Decoder, AutoregressiveWrapper

dataset_addr = "chords"

filez = list()
for dirpath, dirnames, filenames in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
filez

filez.sort()

train_data = torch.Tensor()

for f in tqdm.tqdm(filez):
    train_data = torch.cat(
        (train_data, torch.Tensor(pickle.load(open(f, 'rb'))))
    )
    print('Loaded file:', f)

len(train_data)

len(train_data) / 2048 / 32

train_data[:15], train_data[-15:]

SEQ_LEN = 2048
BATCH_SIZE = 2
NUM_EPOCHS = 100
GRADIENT_ACCUMULATE_EVERY = 1

NUM_BATCHES = (
    len(train_data) // SEQ_LEN // BATCH_SIZE // GRADIENT_ACCUMULATE_EVERY
) * NUM_EPOCHS

LEARNING_RATE = 2e-4

VALIDATE_EVERY = 100
SAVE_EVERY = 1000
GENERATE_EVERY = 200
PRINT_STATS_EVERY = 50

GENERATE_LENGTH = 32

# helpers


def cycle(loader):
    while True:
        for data in loader:
            yield data


# instantiate the model

model = TransformerWrapper(
    num_tokens=3088,
    max_seq_len=SEQ_LEN,
    attn_layers=Decoder(dim=1024, depth=32, heads=8),
)

model = AutoregressiveWrapper(model)

model = torch.nn.DataParallel(model)

# init_model_checkpoint = 'models/Allegro_Music_Transformer_Small_Trained_Model_56000_steps_0.9399_loss_0.7374_acc.pth'
init_model_checkpoint = (
    "models/model_checkpoint_1434_steps_1.552_loss_0.5725_acc.pth"
)


model.load_state_dict(torch.load(init_model_checkpoint))

model.cuda()

summary(model)

# Dataloader


class MusicDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        # random sampling

        idx = secrets.randbelow(self.data.size(0) - self.seq_len - 1)
        full_seq = self.data[idx : idx + self.seq_len + 1].long()

        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0)


train_dataset = MusicDataset(train_data, SEQ_LEN)
val_dataset = MusicDataset(train_data, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_dataset[666]

# Train the model

train_losses = []
val_losses = []

train_accs = []
val_accs = []

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc='Training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss, acc = model(next(train_loader))
        loss.backward(torch.ones(loss.shape).cuda())

    if i % PRINT_STATS_EVERY == 0:
        print(f'Training loss: {loss.mean().item()}')
        print(f'Training acc: {acc.mean().item()}')

    train_losses.append(loss.mean().item())
    train_accs.append(acc.mean().item())

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = model(next(val_loader))

            print(f'Validation loss: {val_loss.mean().item()}')
            print(f'Validation acc: {val_acc.mean().item()}')

            val_losses.append(val_loss.mean().item())
            val_accs.append(val_acc.mean().item())

            print('Plotting training loss graph...')

            tr_loss_list = train_losses
            plt.plot([i for i in range(len(tr_loss_list))], tr_loss_list, 'b')
            plt.show()
            plt.close()
            print('Done!')

            print('Plotting training acc graph...')

            tr_loss_list = train_accs
            plt.plot([i for i in range(len(tr_loss_list))], tr_loss_list, 'b')
            plt.show()
            plt.close()
            print('Done!')

            print('Plotting validation loss graph...')
            tr_loss_list = val_losses
            plt.plot([i for i in range(len(tr_loss_list))], tr_loss_list, 'b')
            plt.show()
            plt.close()
            print('Done!')

            print('Plotting validation acc graph...')
            tr_loss_list = val_accs
            plt.plot([i for i in range(len(tr_loss_list))], tr_loss_list, 'b')
            plt.show()
            plt.close()
            print('Done!')

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]

        print(inp)

        sample = model.module.generate(inp[None, ...], GENERATE_LENGTH)

        print(sample)

    if i % SAVE_EVERY == 0:
        print('Saving model progress. Please wait...')
        print(
            'model_checkpoint_'
            + str(i)
            + '_steps_'
            + str(round(float(train_losses[-1]), 4))
            + '_loss_'
            + str(round(float(train_accs[-1]), 4))
            + '_acc.pth'
        )

        fname = (
            'models/model_checkpoint_'
            + str(i)
            + '_steps_'
            + str(round(float(train_losses[-1]), 4))
            + '_loss_'
            + str(round(float(train_accs[-1]), 4))
            + '_acc.pth'
        )

        torch.save(model.state_dict(), fname)

        data = [train_losses, train_accs, val_losses, val_accs]

        TMIDIX.Tegridy_Any_Pickle_File_Writer(data, 'log/losses_accs')

        print('Done!')


print('Saving model progress. Please wait...')
print(
    'model_checkpoint_'
    + str(i)
    + '_steps_'
    + str(round(float(train_losses[-1]), 4))
    + '_loss_'
    + str(round(float(train_accs[-1]), 4))
    + '_acc.pth'
)

fname = (
    'models/model_checkpoint_'
    + str(i)
    + '_steps_'
    + str(round(float(train_losses[-1]), 4))
    + '_loss_'
    + str(round(float(train_accs[-1]), 4))
    + '_acc.pth'
)

torch.save(model.state_dict(), fname)

# Save training loss graph

plt.plot([i for i in range(len(train_losses))], train_losses, 'b')
plt.savefig('log/training_loss_graph.png')
plt.close()
print('Done!')

# Save training acc graph

plt.plot([i for i in range(len(train_accs))], train_accs, 'b')
plt.savefig('log/training_acc_graph.png')
plt.close()
print('Done!')

# Save validation loss graph

plt.plot([i for i in range(len(val_losses))], val_losses, 'b')
plt.savefig('log/validation_loss_graph.png')
plt.close()
print('Done!')

# Save validation acc graph

plt.plot([i for i in range(len(val_accs))], val_accs, 'b')
plt.savefig('log/validation_acc_graph.png')
plt.close()
print('Done!')

data = [train_losses, train_accs, val_losses, val_accs]

TMIDIX.Tegridy_Any_Pickle_File_Writer(data, 'log/losses_accuracies')
