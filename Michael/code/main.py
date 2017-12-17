# %% Import modules

from dataParser import wavLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# %% Loading data and labels

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

plt.show()







# loop through all folders, import all files in a folder 




# %% Preporcess data: 
#   -> downsample   -> fragmentation    -> extract DWT coeff 
#   -> compute energy index

# %% Run using LSTM

WAVELET_DIM = 64
HIDDEN_DIM = 512
OUTPUT_DIM = 300 // # of speakers
model = LstmWavelet(HIDDEN_DIM, WAVELET_DIM, OUTPUT_DIM)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
 # pre-run score

for _,wavelefCoeff in get_waveletCoeff(file):
    tensor = autograd.Variable(torch.from_numpy(wavelefCoeff))
    tag_scores = model(tensor)
    print(tag_scores)
    break

# train for 6 epoches (70%)
for epoch in range(6):
    for key,wavelefCoeff in get_waveletCoeff(trainFiles): # trainFiles contain all wavelet coeff 
                                                            # for all training dataset
        #if key not in key2num:
        #    continue
        tensor = autograd.Variable(torch.from_numpy(wavelefCoeff))
        tensor = tensor.cuda() # use gpu
        targets = autograd.Variable(torch.from_numpy())
        targets = targets.cuda()

        model.zero_grad()
        model.hidden = model.init_hidden()

        # forward pass
        tag_scores = model(tensor)

        # backprop
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# post-run score (30%)
for _,wavelefCoeff in get_waveletCoeff(testFiles): # testFiles contain all wavelet coeff 
                                                            # for all testing dataset
    tensor = torch.from_numpy(wavelefCoeff)
    tag_scores = model(tensor)
    print(tag_scores)
    break

