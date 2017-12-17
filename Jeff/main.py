from lstm_mfcc import LstmMfcc
import torch
import torch.autograd as autograd
import torch.optim as optim
from kaldi_io import read_mat_scp

HIDDEN_DIM = 512
MFCC_DIM = 20
SPK = 300

def utt2num():
    """create a dictionary that maps utt --> spk --> 1-300"""
    dic = {}

def main():
    model = LstmMfcc(HIDDEN_DIM, MFCC_DIM, SPK)
    model.cuda() # use gpu
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    file = '/export/b17/jlai/michael/mfcc/all_mfcc.scp'

    # pre-run score
    for _,mat in read_mat_scp(file):
        tensor = autograd.Variable(torch.from_numpy(mat))
        tag_scores = model(tensor)
        print(tag_scores)
        break

    # train for 6 epoches (70%)
    for epoch in range(6):
        for key,mat in read_mat_scp(file):
            #if key not in key2num:
            #    continue
            tensor = autograd.Variable(torch.from_numpy(mat))
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
    for _,mat in read_mat_scp(file):
        tensor = torch.from_numpy(mat)
        tag_scores = model(tensor)
        print(tag_scores)
        break

main()
