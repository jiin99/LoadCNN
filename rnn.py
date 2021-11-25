import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import dataloader_simple
from torchsummaryX import summary

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_dim,seq_len,num_layers, dropout=0.5):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(seq_len * hidden_size, output_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        return hidden

    def forward(self, x):
        h_0 = self.init_hidden(x.size(0))
        x, _ = self.rnn(x, h_0)
        #import pdb; pdb.set_trace()
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class LS(nn.Module):

    def __init__(self, input_size, hidden_size, output_dim, seq_len, num_layers,bidirectional = False, dropout=0.5):
        super(LS, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(seq_len * hidden_size, output_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    
    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        return hidden, cell
        
    def forward(self, x):
        h_0, c_0 = self.init_hidden(x.size(0))
        x, _ = self.lstm(x, (h_0, c_0))
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, seq_len, num_layers, bidirectional=False, dropout=0.5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(seq_len * hidden_size * self.num_directions, output_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    
    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).cuda()
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).cuda()
        return hidden, cell
        
    def forward(self, x):
        h_0, c_0 = self.init_hidden(x.size(0))
        x, _ = self.lstm(x, (h_0, c_0))
        import pdb; pdb.set_trace()
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out

if __name__ == "__main__":
    test_dataset = dataloader_simple.TimeSeries_base_test('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',7,'test','rnn')
    inputs = torch.zeros((256,336,1)).cuda()
    model = LSTM(1, 32, 48, 336, 1,bidirectional=False,dropout=0).cuda()
    # model = RNN(1, 32, 48, 336, 5).cuda()
    summary(model, inputs)