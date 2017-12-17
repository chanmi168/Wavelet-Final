import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time_distribute import TimeDistributed

class LstmMfcc(nn.Module):
    def __init__(self, hidden_dim, mfcc_dim, target_dim):
        super(LstmMfcc, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm       = nn.LSTM(input_size=mfcc_dim, hidden_size=hidden_dim,num_layers=2)
        self.hidden2tag = nn.Linear(hidden_dim, target_dim)
        self.hidden     = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))

    def forward(self, mfcc):
        lstm_out, self.hidden = self.lstm(mfcc.view(len(mfcc),1,-1), self.hidden)
        tag_space             = self.hidden2tag(lstm_out.view(len(mfcc), -1))
        score                 = F.log_softmax(tag_space)
        return score
