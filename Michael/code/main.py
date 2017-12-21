# %% Import modules

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lstm_wavelet
import wavelet_dataset

# %% Loading data and labels
"""
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'data/Speaker3/mike_0.wav')
rate, data = wavLoader(filename=my_file)

print(len(data))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(data)



bbb = data[0:300]
print(rate)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(bbb)

# plt.show()

# Calculate wavelet coeff energy
waveletCoeffs = wav2wpc(bbb)




# loop through all folders, import all files in a folder 
"""



# %% Preporcess data: 
#   -> downsample   -> fragmentation    -> extract DWT coeff 
#   -> compute energy index

# %% Run using LSTM

WAVELET_DIM = 20
HIDDEN_DIM = 512
SEQUENCE_DIM = 200
OUTPUT_DIM = 275 # of speakers

train_wavelet_dataset = wavelet_dataset.WaveletDataset(npy_coeff_file='./mixegg.npy', label_file='./mixegg_label.npy')

trainloader = torch.utils.data.DataLoader(train_wavelet_dataset, batch_size=10, shuffle=True, num_workers=2)

test_wavelet_dataset = wavelet_dataset.WaveletDataset(npy_coeff_file='./mixegg.npy', label_file='./mixegg_label.npy')

testloader = torch.utils.data.DataLoader(test_wavelet_dataset, batch_size=10, shuffle=False, num_workers=2)




model = lstm_wavelet.LstmWavelet(HIDDEN_DIM, WAVELET_DIM, SEQUENCE_DIM, OUTPUT_DIM)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# train for 6 epoches (70%)
loss_his = []
print('Start Training')
for epoch in range(30):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        input_coeff, label = data
        
        input_coeff = input_coeff.cuda()
        label = label.cuda()

        input_coeff, label = autograd.Variable(input_coeff.float()), autograd.Variable(label.long())

        model.zero_grad()
        model.hidden = model.init_hidden()

        # forward pass
        result = model(input_coeff)

        # backprop
        loss = loss_function(result, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        #plot
        if i % 5 == 4:
            loss_his.append(running_loss / 5)
            print(running_loss)
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

torch.save(model.state_dict(), './train_weight.pt')
print('Finished Training')


# post-run score (30%)
for i, data in enumerate(testloader, 0):
        
        input_coeff, label = data
        
        input_coeff = input_coeff.cuda()
        label = label.cuda()

        input_coeff, label = autograd.Variable(input_coeff.float()), autograd.Variable(label.long())

        model.hidden = model.init_hidden()

        # forward pass
        result = model(input_coeff)
        result1 = result.data.cpu().numpy()
        speaker = result1.argmax(axis=1)
        print(speaker)
        

