# %% Import modules

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import lstm_wavelet
import wavelet_dataset

# %% Run using LSTM

WAVELET_DIM = 16
HIDDEN_DIM = 512
SEQUENCE_DIM = 200
OUTPUT_DIM = 275 # of speakers

train_wavelet_dataset = wavelet_dataset.WaveletDataset(npy_coeff_file='./coeff/trainDataset.npy', label_file='./coeff/trainLabels.npy')

trainloader = torch.utils.data.DataLoader(train_wavelet_dataset, batch_size=10, shuffle=True, num_workers=2)

test_wavelet_dataset = wavelet_dataset.WaveletDataset(npy_coeff_file='./coeff/testDataset.npy', label_file='./coeff/testLabels.npy')

testloader = torch.utils.data.DataLoader(test_wavelet_dataset, batch_size=10, shuffle=False, num_workers=2)




model = lstm_wavelet.LstmWavelet(HIDDEN_DIM, WAVELET_DIM, SEQUENCE_DIM, OUTPUT_DIM)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.000000001)


# train for 6 epoches (70%)
loss_his = []
print('Start Training')
for epoch in range(10):

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
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

torch.save(model.state_dict(), './train_weight.pt')
print('Finished Training')

print('Start Testing')
# post-run score (30%)
right_num = 0
total_num = test_wavelet_dataset.__len__()
out = open('./output.txt', 'w')
for i, data in enumerate(testloader, 0):
    
    input_coeff, label = data
    
    input_coeff = input_coeff.cuda()
    label = label.cuda()

    input_coeff, label = autograd.Variable(input_coeff.float()), autograd.Variable(label.long())

    model.hidden = model.init_hidden()

    # forward pass
    result = model(input_coeff)
    result1 = result.data.cpu().numpy()
    label1 = label.data.cpu().numpy()
    speaker = result1.argmax(axis=1)
    for n in range(len(speaker)):
        if int(speaker[n]) == int(label1[n]):
            right_num += 1
        out.write("%i " % speaker[n])
    out.write('\n')
out.close()
accuracy = float(right_num)/total_num
print ('The accuracy is %.5f %%' % (accuracy*100))
print('Finished Testing')
