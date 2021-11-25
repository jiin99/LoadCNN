# from rnn import RNN,LSTM
import cnn_simple 
import rnn
import dataloader_simple
from dataloader_simple import TimeSeries,TimeSeries_base_test
import util
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import time
import argparse
import pandas as pd
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import wandb
import glob
# wandb.init(project="time_ensemble")
is_best = 0.65

parser = argparse.ArgumentParser(description='PyTorch code: ODIN')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
# parser.add_argument('--save_root', default = './ensemble',required=True, help='log save path')
# parser.add_argument('--network', type=str, default='dense_C',  required=True, help='dense | res | vgg16')
# parser.add_argument('--pre_trained_net', default='ckpt', help='names')
parser.add_argument('--gpu', type=str, default='7', help='gpu index')
# parser.add_argument('--days',type = str, default = '7', help ='data')
# parser.add_argument('--tries',type = str, default = '1', required = True, help ='try')
args = parser.parse_args()


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self,yhat,y):
        mse_value = self.mse(yhat,y).mean(axis=1) + 1e-8
        return torch.sqrt(mse_value).mean()
class NRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self,yhat,y):
        mse_value = self.mse(yhat,y).mean(axis=1) + 1e-8
        return torch.sqrt(mse_value).mean() / (torch.max(y) - torch.min(y))  

def inverse_10(x):
  return x * 10

def main():
    global is_best

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    test_logger = util.Logger('plot_testing' + '/plot.log')
    li = []
    # criterion = nn.L1Loss()#RMSELoss().cuda()
    # test_dataset = TimeSeries_base_test('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',7,'test','cnn')
    # test_loader = DataLoader(test_dataset,shuffle=False,batch_size = 256,drop_last = False)
    # model = cnn_simple.CNN_big(7)
    # model = rnn.LS(1, 32, 48, 336, 1, bidirectional=False,dropout=0).cuda()
    # model = rnn.RNN(1, 32, 48, 336,1,dropout=0).cuda()
    # model = rnn.LSTM(1, 32, 48, 336, 3, bidirectional=True).cuda()
    # pre_trained_net = '/daintlab/home/jiin9/LoadCNN/bi_lstm_3_adam_real/7days1/ckpt_epoch30.pt'
    # model.load_state_dict(torch.load(pre_trained_net))
    # test_prec = validate(test_loader,model,criterion,'test',li)
    # log = np.concatenate(li)
    # np.save('./log/bi_lstm5', log)

    # print(test_prec)
    # print('test prec : ',test_prec)


    criterion = [RMSELoss().cuda(),NRMSELoss().cuda(),torch.nn.L1Loss()]
    for i in criterion:
        test_dataset = TimeSeries_base_test('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',7,'test','rnn')
        test_loader = DataLoader(test_dataset,shuffle=False,batch_size = 256,drop_last = False)
        # model = cnn_simple.CNN_big(7)
        model = rnn.LSTM(1, 32, 48, 336, 3, bidirectional=True).cuda()
        # model = rnn.RNN(1, 32, 48, 336,5).cuda()
        # model = rnn.LS(1, 32, 48, 336, 5, bidirectional=False).cuda()
        pre_trained_net = '/daintlab/home/jiin9/LoadCNN/bi_lstm_3_adam_real/7days1/ckpt_epoch30.pt'
        model.load_state_dict(torch.load(pre_trained_net))
        test_prec = validate(test_loader,model,i,'test',li)
        print(test_prec)
        print(i)
        print('test prec : ',test_prec)


def validate(val_loader,model,criterion,mode,test_logger):
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()

    #switch to evaluate mode
    model.eval()
    with torch.no_grad():
        val_loss = 0
        end = time.time()
        
        for i, (val_x,val_y) in tqdm(enumerate(val_loader)):
            # val_x, val_y = val_x.unsqueeze(1).float().cuda(), val_y.squeeze().float().cuda()
            val_x, val_y = val_x.float().cuda(), val_y.squeeze().float().cuda()
            
            val_output = model(val_x)

            val_output = inverse_10(val_output)
            val_y = inverse_10(val_y)
            #v_loss = torch.sqrt(criterion(val_output,val_y)) --rmse

            v_loss = criterion(val_output,val_y)
            losses.update(v_loss.item(), val_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            val_loss += v_loss.item()
            
            if i % 100 == 0:
                print(mode, ': [{0}/{1}]\t'
                      'V_Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'V_Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                                i, len(val_loader), batch_time=batch_time, loss=losses))
            # import pdb; pdb.set_trace()
            test_logger.append(val_output.reshape(-1,1).detach().cpu().numpy())
        plt.figure(figsize = (20,5))
        # import pdb; pdb.set_trace()
        # plt.plot(val_output.detach().cpu().numpy()[5],linestyle = '--' )
        plt.plot(val_y.detach().cpu().numpy()[1],color = 'black',linewidth = 5)
        plt.savefig('valid3.png', dpi=300)
        
    return losses.avg
if __name__ == "__main__":
    main()