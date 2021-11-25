import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler


def scaling_10(x):
  return x / 10
def inverse_10(x):
  return x * 10


def one_hot_embedding(labels, num_classes):
    y = np.eye(num_classes)
    return y[labels]

def multi_hot_embedding(labels):
    y = np.zeros((2,31))
    li = []
    for i in range(len(labels)):
        p = i//31
        pp = i% 31
        y[0][p] = 1
        y[1][pp] = 1
        li.append(y.tolist())
        y=np.zeros((2,31))
    return np.array(li)

def one_hot_encoding(data_y):
    #print("one_hot_encoding process")
    cls = set(data_y)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, data_y)))
    return one_hot

class TimeSeries(Dataset):
    def __init__(self, data_frame,days,types):
        data = pd.read_csv(data_frame, index_col=0)
        data['Usage'] = scaling_10(data['Usage'])
        self.data = data.values
        self.length = days
        self.output_len = (days *48) // 9
        df = pd.read_csv('/daintlab/data/CER_electricity/'+ str(days)+'days'+types+'_vari.csv')
        self.before_index = df.values[:,1].tolist()
    def __len__(self):
        return len(self.before_index)

    def __getitem__(self, index):

        idx = self.before_index[index]

        x = self.data[idx : idx + self.length * 48, 1].reshape(1,-1)
        y = self.data[idx + self.length * 48 : idx + self.length * 48 + self.output_len, 1].reshape(1, self.output_len)

        return x, y

class TimeSeries_base(Dataset):
    def __init__(self, data_frame,days,types):
        data = pd.read_csv(data_frame, index_col=0)
        data['Usage'] = scaling_10(data['Usage'])
        self.data = data.values
        self.length = days

        df = pd.read_csv('/daintlab/data/CER_electricity/'+ str(days)+'days'+types+'_base.csv')
        self.before_index = df.values[:,1].tolist()
        print('length : ',len(self.before_index))
        
    def __len__(self):
        return len(self.before_index)

    def __getitem__(self, index):

        idx = self.before_index[index]################### 
        x = self.data[idx : idx + self.length * 48, 1].reshape(-1,1)
        y = self.data[idx + self.length * 48 : idx + self.length * 48 + 48, 1].reshape(-1,1)
        return x, y

class TimeSeries_diff_out(Dataset):
    def __init__(self, data_frame,days,types,tries):
        data = pd.read_csv(data_frame, index_col=0)
        data['Usage'] = scaling_10(data['Usage'])
        self.data = data.values
        self.length = days
        self.tries = tries

        df = pd.read_csv('/daintlab/data/CER_electricity/'+ str(days)+'days'+types+'_base.csv')
        self.before_index = df.values[:,1].tolist()
        print('length : ',len(self.before_index))
        
    def __len__(self):
        return len(self.before_index)

    def __getitem__(self, index):

        idx = self.before_index[index]

        x = self.data[idx : idx + self.length * 48, 1].reshape(1,-1)
        y = self.data[idx + self.length * 48 + 12*self.tries : idx + self.length * 48 + 12*(self.tries+1), 1].reshape(1,-1)
        return x, y

class TimeSeries_diff_in(Dataset):
    def __init__(self, data_frame,days,types,tries):
        data = pd.read_csv(data_frame, index_col=0)
        data['Usage'] = scaling_10(data['Usage'])
        self.data = data.values
        self.length = days
        self.tries = tries

        df = pd.read_csv('/daintlab/data/CER_electricity/'+ str(days)+'days'+types+'_base.csv')
        self.before_index = df.values[:,1].tolist()
        print('length : ',len(self.before_index))
        
    def __len__(self):
        return len(self.before_index)

    def __getitem__(self, index):

        idx = self.before_index[index]

        arr = []
        for i in range(7):
            arr.append(self.data[idx + i*48 : idx + (i+1) * 48, 1][12*self.tries : 12*(self.tries +1)])
        x = np.concatenate(arr).reshape(1,-1)
        y = self.data[idx + self.length * 48 + 12*self.tries : idx + self.length * 48 + 12*(self.tries+1), 1].reshape(1,-1)
        return x, y


class TimeSeries_base_test(Dataset):
    def __init__(self, data_frame,days,types,model):
        data = pd.read_csv(data_frame, index_col=0)
        data['Usage'] = scaling_10(data['Usage'])
        self.data = data.values
        self.length = days

        df = pd.read_csv('/daintlab/data/CER_electricity/'+ str(days)+'days'+types+'_base.csv')
        self.before_index = df.values[:,1].tolist()
        print('length : ',len(self.before_index))
        self.model = model
        
    def __len__(self):
        return len(self.before_index)

    def __getitem__(self, index):

        idx = self.before_index[index]
        if self.model == 'cnn':
            x = self.data[idx : idx + self.length * 48, 1].reshape(1,-1)
            y = self.data[idx + self.length * 48 : idx + self.length * 48 + 48, 1].reshape(1,-1)
        else : 
            x = self.data[idx : idx + self.length * 48, 1].reshape(-1,1)
            y = self.data[idx + self.length * 48 : idx + self.length * 48 + 48, 1].reshape(-1,1)
        return x, y

#diffrent length of output
def data_set(days,batch):

    train_dataset = TimeSeries('/daintlab/data/CER_electricity/the_end_train_onehot.csv',days,'train')
    valid_dataset = TimeSeries('/daintlab/data/CER_electricity/the_end_valid_onehot.csv',days,'valid')
    test_dataset = TimeSeries('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',days,'test')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)
    valid_loader = DataLoader(valid_dataset,shuffle=False,batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)

    return train_loader, test_loader, valid_loader

#fixed length of output
def data_set_base(days,batch):

    train_dataset = TimeSeries_base('/daintlab/data/CER_electricity/the_end_train_onehot.csv',days,'train')
    valid_dataset = TimeSeries_base('/daintlab/data/CER_electricity/the_end_valid_onehot.csv',days,'valid')
    test_dataset = TimeSeries_base('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',days,'test')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)
    valid_loader = DataLoader(valid_dataset,shuffle=False,batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)

    return train_loader, test_loader, valid_loader

def data_set_diff_out(days,batch,tries):

    train_dataset = TimeSeries_diff_out('/daintlab/data/CER_electricity/the_end_train_onehot.csv',days,'train',tries)
    valid_dataset = TimeSeries_diff_out('/daintlab/data/CER_electricity/the_end_valid_onehot.csv',days,'valid',tries)
    test_dataset = TimeSeries_diff_out('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',days,'test',tries)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)
    valid_loader = DataLoader(valid_dataset,shuffle=False,batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)

    return train_loader, test_loader, valid_loader

def data_set_diff_in(days,batch,tries):

    train_dataset = TimeSeries_diff_in('/daintlab/data/CER_electricity/the_end_train_onehot.csv',days,'train',tries)
    valid_dataset = TimeSeries_diff_in('/daintlab/data/CER_electricity/the_end_valid_onehot.csv',days,'valid',tries)
    test_dataset = TimeSeries_diff_in('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',days,'test',tries)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)
    valid_loader = DataLoader(valid_dataset,shuffle=False,batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size = batch, pin_memory = True,drop_last = True, num_workers=2)

    return train_loader, test_loader, valid_loader