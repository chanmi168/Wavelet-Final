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

WAVELET_DIM = 8
HIDDEN_DIM = 512
SEQUENCE_DIM = 200
OUTPUT_DIM = 275 # of speakers

train_wavelet_dataset = wavelet_dataset.WaveletDataset(npy_coeff_file='./coeff/trainDatasetNormed.npy', label_file='./coeff/trainLabels.npy')

trainloader = torch.utils.data.DataLoader(train_wavelet_dataset, batch_size=10, shuffle=True, num_workers=2)

train_t_wavelet_dataset = wavelet_dataset.WaveletDataset(npy_coeff_file='./coeff/trainDatasetNormed.npy', label_file='./coeff/trainLabels.npy')

train_t_loader = torch.utils.data.DataLoader(train_t_wavelet_dataset, batch_size=10, shuffle=False, num_workers=2)

test_wavelet_dataset = wavelet_dataset.WaveletDataset(npy_coeff_file='./coeff/testDatasetNormed.npy', label_file='./coeff/testLabels.npy')

testloader = torch.utils.data.DataLoader(test_wavelet_dataset, batch_size=10, shuffle=False, num_workers=2)

learning_rate = 0.08
epoch_num = 1
train_test = 0
trainaddtest = 1
model = lstm_wavelet.LstmWavelet(HIDDEN_DIM, WAVELET_DIM, SEQUENCE_DIM, OUTPUT_DIM)
loss_function = nn.NLLLoss()
#optimizer = optim.Adam(model.parameters(),lr=0.05)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

if train_test == 0:
    # train for 6 epoches (70%)
    loss_his = []
    epoch_end =[]
    print('Start Training')
    for epoch in range(epoch_num):
    
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
            
            if i == len(trainloader)-1:
                epoch_end.append(loss.data[0])
                print('end_epoch loss:'+str(loss.data[0]))
                print(' ')
            #plot
            if i % 100 == 99:    # print every 2000 mini-batches
                loss_his.append(running_loss / 100)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        torch.save(model.state_dict(), './pt/'+ str(HIDDEN_DIM) + '_' + str(learning_rate) +'_train_weight_'+ str(epoch+1) +'.pt')
    fig=plt.figure()
    plt.plot(loss_his)
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.show()
    fig.savefig('./output_'+ str(learning_rate) + '_' + str(epoch_num) +'.png')
    epoch_name = './epoch_'+ str(learning_rate) + '_' + str(epoch_num) +'.txt'
    epoch_file = open(epoch_name, 'w')
    for item in range(len(epoch_end)):
        epoch_file.write('epoch '+ str(item) +' loss: ' + str(epoch_end[item])+ '\n')
    epoch_file.close()
    print('Finished Training')
else:
    model.load_state_dict(torch.load('./pt/0.08_train_weight_30.pt'))
    print('Start Testing')
    print('learning rate: ' +str(learning_rate))
    # post-run score (30%)
    right_num = 0
    total_num = train_t_wavelet_dataset.__len__()
    for i, data in enumerate(train_t_loader, 0):
        
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
                
    accuracy = float(right_num)/total_num
    print('The train accuracy is %.5f %%' % (accuracy*100))
    
    
    overall_right_num = 0
    total_num = test_wavelet_dataset.__len__()
    right_num = 0
    num = 0
    who = 0
    out = open('./output_'+ str(learning_rate) + '_' + str(epoch_num) +'.txt', 'w')
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
        if who == int(label1[0]):
            for n in range(len(speaker)):
                num += 1
                if int(speaker[n]) == int(label1[n]):
                    overall_right_num += 1
                    right_num += 1
                out.write("%i " % speaker[n])
            out.write('\n')
        else:
            temp_acc = float(right_num)/num
            out.write('Label ' + str(who) + ' accuracy:' +str(temp_acc*100) +'%\n\n')
            who = int(label1[0])
            right_num = 0
            num = 0
            for n in range(len(speaker)):
                num += 1
                if int(speaker[n]) == int(label1[n]):
                    overall_right_num += 1
                    right_num += 1
                out.write("%i " % speaker[n])
            out.write('\n')
        
    out.close()
    accuracy = float(overall_right_num)/total_num
    print('The test accuracy is %.5f %%' % (accuracy*100))
    print('Finished Testing')
if trainaddtest == 1:
    print('Start Testing')
    print('learning rate: ' +str(learning_rate))
    # post-run score (30%)
    right_num = 0
    total_num = train_t_wavelet_dataset.__len__()
    for i, data in enumerate(train_t_loader, 0):
        
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
                
    accuracy = float(right_num)/total_num
    print('The train accuracy is %.5f %%' % (accuracy*100))
    
    
    overall_right_num = 0
    total_num = test_wavelet_dataset.__len__()
    right_num = 0
    num = 0
    who = 0
    out = open('./output_'+ str(learning_rate) + '_' + str(epoch_num) +'.txt', 'w')
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
        if who == int(label1[0]):
            for n in range(len(speaker)):
                num += 1
                if int(speaker[n]) == int(label1[n]):
                    overall_right_num += 1
                    right_num += 1
                out.write("%i " % speaker[n])
            out.write('\n')
        else:
            temp_acc = float(right_num)/num
            out.write('Label ' + str(who) + ' accuracy:' +str(temp_acc*100) +'%\n\n')
            who = int(label1[0])
            right_num = 0
            num = 0
            for n in range(len(speaker)):
                num += 1
                if int(speaker[n]) == int(label1[n]):
                    overall_right_num += 1
                    right_num += 1
                out.write("%i " % speaker[n])
            out.write('\n')
        
    out.close()
    accuracy = float(overall_right_num)/total_num
    print('The test accuracy is %.5f %%' % (accuracy*100))
    print('Finished Testing')