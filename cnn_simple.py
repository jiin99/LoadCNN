import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from dataloader_simple import TimeSeries,TimeSeries_base
import os
import dataloader_simple


#different length of output
class CNN(nn.Module):
    def __init__(self,input_days):
        super(CNN, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=(1,7),padding = (0,3)),
            nn.LeakyReLU(),

            nn.Conv2d(16,24,kernel_size=(1,5),padding = (0,4)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(24,24,kernel_size=(1,5),padding = (0,2)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(24,64,kernel_size=(1,4),padding = (0,2)),
            nn.LeakyReLU(),

            nn.Conv2d(64,64,kernel_size=(1,3),padding = 0),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(64,64,kernel_size=(1,3),padding = (0,1)),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(

            nn.Linear(input_days*64*6,(input_days*48)//9),

        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out

#fixed length of output
class CNN_base(nn.Module):
    def __init__(self,input_days):
        super(CNN_base, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=(1,7),padding = (0,3)),
            nn.LeakyReLU(),

            nn.Conv2d(16,24,kernel_size=(1,5),padding = (0,4)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(24,24,kernel_size=(1,5),padding = (0,2)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(24,64,kernel_size=(1,4),padding = (0,2)),
            nn.LeakyReLU(),

            nn.Conv2d(64,64,kernel_size=(1,3),padding = 0),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(64,64,kernel_size=(1,3),padding = (0,1)),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(

            nn.Linear(input_days*64*6,48),

        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out

class CNN_small(nn.Module):
    def __init__(self,input_days):
        super(CNN_small, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,8, kernel_size=(1,7),padding = (0,3)),
            nn.LeakyReLU(),
        
            nn.Conv2d(8,16,kernel_size=(1,5),padding = (0,4)),
            nn.LeakyReLU(),
            
            nn.MaxPool2d(kernel_size = (1,2),padding = 0),
            
            nn.Conv2d(16,16,kernel_size=(1,5),padding = (0,2)),
            nn.LeakyReLU(),
            
            nn.MaxPool2d(kernel_size = (1,2),padding = 0),
        
            nn.Conv2d(16,32,kernel_size=(1,4),padding = (0,2)),
            nn.LeakyReLU(),
            
            nn.Conv2d(32,32,kernel_size=(1,3),padding = 0),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),
            
            nn.Conv2d(32,32,kernel_size=(1,3),padding = (0,1)),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(
            
            nn.Linear(input_days*32*6,48),
            
        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()
    
    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out

class CNN_big_rev(nn.Module):
    def __init__(self,input_days):
        super(CNN_big_rev, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(7,1),padding = (3,0)),
            nn.LeakyReLU(),

            nn.Conv2d(32,32,kernel_size=(5,1),padding = (4,0)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (2,1),padding = 0),

            nn.Conv2d(32,64,kernel_size=(5,1),padding = (2,0)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (2,1),padding = 0),

            nn.Conv2d(64,128,kernel_size=(4,1),padding = (2,0)),
            nn.LeakyReLU(),

            nn.Conv2d(128,128,kernel_size=(3,1),padding = 0),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (2,1),padding = 0),

            nn.Conv2d(128,256,kernel_size=(3,1),padding = (1,0)),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(

            nn.Linear(input_days*256*6,48),

        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out

class CNN_big(nn.Module):
    def __init__(self,input_days):
        super(CNN_big, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(1,7),padding = (0,3)),
            nn.LeakyReLU(),

            nn.Conv2d(32,32,kernel_size=(1,5),padding = (0,4)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(32,64,kernel_size=(1,5),padding = (0,2)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(64,128,kernel_size=(1,4),padding = (0,2)),
            nn.LeakyReLU(),

            nn.Conv2d(128,128,kernel_size=(1,3),padding = 0),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(128,256,kernel_size=(1,3),padding = (0,1)),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(

            nn.Linear(input_days*256*6,48),

        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out

class CNN_big_diff(nn.Module):
    def __init__(self,input_days):
        super(CNN_big_diff, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(1,7),padding = (0,3)),
            nn.LeakyReLU(),

            nn.Conv2d(32,32,kernel_size=(1,5),padding = (0,4)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(32,64,kernel_size=(1,5),padding = (0,2)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(64,128,kernel_size=(1,4),padding = (0,2)),
            nn.LeakyReLU(),

            nn.Conv2d(128,128,kernel_size=(1,3),padding = 0),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(128,256,kernel_size=(1,3),padding = (0,1)),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(

            nn.Linear(input_days*256*6,12),

        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out

class CNN_big_vari(nn.Module):
    def __init__(self,input_days):
        super(CNN_big_vari, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(1,7),padding = (0,3)),
            nn.LeakyReLU(),

            nn.Conv2d(32,32,kernel_size=(1,5),padding = (0,4)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(32,64,kernel_size=(1,5),padding = (0,2)),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(64,128,kernel_size=(1,4),padding = (0,2)),
            nn.LeakyReLU(),

            nn.Conv2d(128,128,kernel_size=(1,3),padding = 0),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(128,256,kernel_size=(1,3),padding = (0,1)),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(

            nn.Linear(input_days*256*6,(input_days*48)//9),

        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out
class CNN_new(nn.Module):
    def __init__(self,input_days):
        super(CNN_new, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(1,7),padding = (0,3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32,32,kernel_size=(1,5),padding = (0,4)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(32,64,kernel_size=(1,5),padding = (0,2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(64,128,kernel_size=(1,4),padding = (0,2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128,128,kernel_size=(1,3),padding = 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(128,256,kernel_size=(1,3),padding = (0,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(

            nn.Linear(input_days*256*6,48),
            #nn.ReLU()
        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out

class CNN_new_vari(nn.Module):
    def __init__(self,input_days):
        super(CNN_new_vari, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(1,7),padding = (0,3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32,32,kernel_size=(1,5),padding = (0,4)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(32,64,kernel_size=(1,5),padding = (0,2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(64,128,kernel_size=(1,4),padding = (0,2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128,128,kernel_size=(1,3),padding = 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(128,256,kernel_size=(1,3),padding = (0,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(

            nn.Linear(input_days*256*6,(input_days*48)//9),
            #nn.ReLU()
        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out

class CNN_new2(nn.Module):
    def __init__(self,input_days):
        super(CNN_new2, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=(1,7),padding = (0,3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64,64,kernel_size=(1,5),padding = (0,4)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(64,128,kernel_size=(1,5),padding = (0,2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(128,256,kernel_size=(1,4),padding = (0,2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256,256,kernel_size=(1,3),padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size = (1,2),padding = 0),

            nn.Conv2d(256,512,kernel_size=(1,3),padding = (0,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.fc_module = nn.Sequential(

            nn.Linear(input_days*256*6,48),
            #nn.ReLU()
        )
        if torch.cuda.is_available():
            self.cnn_module = self.cnn_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self,x):
        out = self.cnn_module(x)
        out = out.view(out.size(0),-1)
        out = self.fc_module(out)

        return out

# day = [1,3,5,7,9]
# for i in day : 
#     print('day : ',i)
#     test_dataset = TimeSeries('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',i,'test')
#     test_loader = DataLoader(test_dataset,shuffle=False,batch_size = 64,drop_last = False)
#     x,y = next(iter(test_loader))
#     # import pdb; pdb.set_trace()
#     x = x.unsqueeze(1).float().cuda()
#     my_net = CNN_new_vari(i)
#     summary(model = my_net.cuda(),input_size = x.size()[1:])


# if __name__ == '__main__':
#     from torchsummary import summary
#     my_net = ae_acnn(in_shape=(1, 256, 256))
#     summary(model=my_net.cuda(), input_size=(1, 256, 256))

# test_dataset = TimeSeries('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',7,'test')
# test_loader = DataLoader(test_dataset,shuffle=False,batch_size = 64,drop_last = False)
# x,y = next(iter(test_loader))
# # import pdb; pdb.set_trace()
# x = x.unsqueeze(1).float().cuda()
# my_net = CNN_big(7)
# summary(model = my_net.cuda(),input_size = x.size()[1:])
