# Implementation of LSTM

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time_distribute import TimeDistributed

# LSTM MODEL
class LstmWavelet(nn.Module):
    
    def __init__(self, hidden_dim, wavelet_dim, sequence_dim, target_dim):
        super(LstmWavelet, self).__init__()
        
        # Define parameters of model
        self.hidden_dim     = hidden_dim
        self.wavelet_dim    = wavelet_dim
        self.sequence_dim   = sequence_dim
        self.target_dim     = target_dim

        # Define layers
        self.lstm       = nn.LSTM(input_size=self.wavelet_dim, hidden_size=self.hidden_dim, num_layers=2, batch_first=True)
        self.linear1    = nn.Linear(self.hidden_dim, self.target_dim)
        self.linear2    = nn.Linear(self.sequence_dim, 1)

        # Define initial hidden state
        self.hidden     = self.init_hidden()

    def init_hidden(self):

        # zero initial hidden state
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(2, 10, self.hidden_dim)),
                autograd.Variable(torch.zeros(2, 10, self.hidden_dim)))

    def forward(self, wavelet):
        #wavelet size: 10 X 200 X 64
        
        lstm_out, self.hidden = self.lstm(wavelet, self.hidden)
        #lstm_out size: 10 X 200 X 512
    
        candidate_space       = self.linear1(lstm_out)
        #candidate_space size: 10 X 200 X 275
        
        #transpose 10 X 275 X 200
        shrinked              = self.linear2(torch.transpose(candidate_space, 1, 2))
        #shrinked size: 10 X 275 X 1
        
        #view 10 X 275
        result                = F.log_softmax(shrinked.view(10, -1))
        return result
