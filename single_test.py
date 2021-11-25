# from rnn import RNN,LSTM
import cnn_simple 
import rnn
import dataloader_simple
from dataloader_simple import TimeSeries,TimeSeries_base
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
import random
# wandb.init(project="time_ensemble")
is_best = 0.65

parser = argparse.ArgumentParser(description='PyTorch code: ODIN')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
# parser.add_argument('--save_root', default = './ensemble',required=True, help='log save path')
# parser.add_argument('--network', type=str, default='dense_C',  required=True, help='dense | res | vgg16')
# parser.add_argument('--pre_trained_net', default='ckpt', help='names')
parser.add_argument('--gpu', type=str, default='0', help='gpu index')
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


#mae loss
#torch.nn.L1Loss()

def inverse_10(x):
  return x * 10

def main():
    global is_best

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    

    criterion = nn.L1Loss().cuda()#NRMSELoss().cuda()
    directory = ['1days_1/best/ckpt_best30.pt','3days_1/best/ckpt_best30.pt','5days_1/best/ckpt_best30.pt','7days_1/best/ckpt_best30.pt','9days_1/best/ckpt_best30.pt']
    for tries,i in enumerate(directory):
        day = tries*2 +1
        test_dataset = TimeSeries('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',day,'test')
        test_loader = DataLoader(test_dataset,shuffle=False,batch_size = 256,drop_last = False)
        model = cnn_simple.CNN_big_vari(day).cuda()
        pre_trained_net = '/daintlab/home/jiin9/LoadCNN/ensemble_big_vari/' + i
        model.load_state_dict(torch.load(pre_trained_net))
        test_prec = validate(test_loader,model,criterion,'test')
        print(test_prec)
        print('rmse single prec : ',test_prec)
    
    # directory = ['paper_smaller_cnn','paper_big_cnn','lstm_1_adam','lstm_3_adam','lstm_5_adam','rnn_1_adam','rnn_3_adam','rnn_5_adam','bi_lstm_3_adam','bi_lstm_5_adam']
    # model = [cnn_simple.CNN_small(7).cuda(),cnn_simple.CNN_big(7).cuda(),rnn.LS(1, 32, 48, 336, 1, bidirectional=False,dropout=0).cuda(),rnn.LS(1, 32, 48, 336, 3, bidirectional=False).cuda()
    # ,rnn.LS(1, 32, 48, 336, 5, bidirectional=False).cuda(),rnn.RNN(1, 32, 48, 336, 1,dropout = 0).cuda(),rnn.RNN(1, 32, 48, 336, 3).cuda(),rnn.RNN(1, 32, 48, 336, 5).cuda()
    # ,rnn.LS(1, 32, 48, 336, 3, bidirectional=True).cuda(),rnn.LS(1, 32, 48, 336, 5, bidirectional=True).cuda()]
    # for num,dir in enumerate(directory):
        # i = (num*2)+1
        # print('days :',i)
        # if num <3:
        #     test_dataset = TimeSeries_base('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',7,'test','cnn')
        #     test_loader = DataLoader(test_dataset,shuffle=False,batch_size = 256,drop_last = False)
        #     model = model[num]
        #     pre_trained_net = '/daintlab/home/jiin9/LoadCNN/' + dir +'/7days1/ckpt_epoch30.pt'
        #     model.load_state_dict(torch.load(pre_trained_net))
        #     test_prec = validate(test_loader,model,criterion,'test',dir,'cnn')
        # else : 
        #     test_dataset = TimeSeries_base('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',7,'test','rnn')
        #     test_loader = DataLoader(test_dataset,shuffle=False,batch_size = 256,drop_last = False)
        #     model = model[num]
        #     pre_trained_net = '/daintlab/home/jiin9/LoadCNN/' + dir +'/7days1/ckpt_epoch30.pt'
        #     model.load_state_dict(torch.load(pre_trained_net))
        #     test_prec = validate(test_loader,model,criterion,'test',dir)
        # plt.close()
        # print(test_prec)
        # print('rmse test prec : ',test_prec)
def validate(val_loader,model,criterion,mode):
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
            val_x, val_y = val_x.unsqueeze(1).float().cuda(), val_y.squeeze().float().cuda()
            # val_x, val_y = val_x.float().cuda(), val_y.squeeze().float().cuda()
            
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
        
    return losses.avg

def validate_cnn(val_loader,model,criterion,mode,dir,arch):
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
            if arch == 'cnn':
                val_x, val_y = val_x.unsqueeze(1).float().cuda(), val_y.squeeze().float().cuda()
            else : 
                val_x = val_x.float().cuda()
                val_y = val_y.squeeze().float().cuda()
            
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
    for i in range(0,10):
        plt.plot(val_y.detach().cpu().numpy()[i],'r',label = 'Real Usage')
        plt.plot(val_output.detach().cpu().numpy()[i], linestyle='--', label = dir)
        plt.xlabel('time(30min)')
        plt.ylabel('electricity usage(kWh)')    
        plt.savefig(f'images_random{i}.png', dpi=300)
          
    return losses.avg

def validate_rnn(val_loader,model,criterion,mode,dir):
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
            val_x = val_x.float().cuda()
            val_y = val_y.squeeze().float().cuda()
            
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
    for i in range(0,30):
        plt.plot(val_y.squeeze(1)[i].detach().cpu().numpy(),'r',label = 'Real Usage')
        plt.plot(val_output.detach().cpu().numpy()[i], linestyle='--', label = dir)
        plt.xlabel('time(30min)')
        plt.ylabel('electricity usage(kWh)')    
        plt.savefig(f'images_random{i}.png', dpi=300)
          
    return losses.avg
if __name__ == "__main__":
    main()