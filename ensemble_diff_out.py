'''
input :7,7,7,7,7days, output : 48step
'''

# from cnn import CNN_3days,CNN_7days,CNN_BN_3days,CNN_BN_7days,CNN_GN_3days,CNN_GN_7days,CNN_3days_half,CNN_7days_half
# from rnn import RNN,LSTM
import cnn_simple 
import dataloader_simple
from dataloader_simple import TimeSeries,TimeSeries_base,TimeSeries_diff_out
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
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for data loader')
# parser.add_argument('--save_root', default = './ensemble',required=True, help='log save path')
# parser.add_argument('--network', type=str, default='dense_C',  required=True, help='dense | res | vgg16')
# parser.add_argument('--pre_trained_net', default='ckpt', help='names')
parser.add_argument('--gpu', type=str, default='2', help='gpu index')
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

    # criterion = NRMSELoss().cuda()
    criterion = nn.L1Loss()
    test_dataset =  TimeSeries_base('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',7,'test')
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size = args.batch_size,drop_last = False)
    directory = ['7days0/ckpt_epoch30.pt','7days1/ckpt_epoch30.pt','7days2/ckpt_epoch30.pt','7days3/ckpt_epoch30.pt']
    test_prec = validate(test_loader,criterion,directory,'test')
    print(test_prec)
    print(' ensemble rmse test prec : ',test_prec)
    model = cnn_simple.CNN_big_diff(7)
    for tries,i in enumerate(directory):
        test_dataset =  test_dataset = TimeSeries_diff_out('/daintlab/data/CER_electricity/the_end_test_onehot_del2.csv',7,'test',tries)
        test_loader = DataLoader(test_dataset,shuffle=False,batch_size = args.batch_size,drop_last = False)
        pre_trained_net = '/daintlab/home/jiin9/LoadCNN/cnn_diff_out_ensemble/' + i
        model.load_state_dict(torch.load(pre_trained_net))
        test_prec = validate_single(test_loader,model,criterion,'test')
        print(test_prec)
        print('rmse single prec : ',test_prec)
    
    



def validate(val_loader,criterion,directory,mode):
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    
    with torch.no_grad():
        val_loss = 0
        end = time.time()
        
        for i, (val_x,val_y) in tqdm(enumerate(val_loader)):
            val_x, val_y = val_x.unsqueeze(1).float().cuda(), val_y.squeeze().float().cuda()
            all_output = []
            for p,k in enumerate(directory) :
                model = cnn_simple.CNN_big_diff(7)
                pre_trained_net = '/daintlab/home/jiin9/LoadCNN/cnn_diff_out_ensemble/' + k
                model.load_state_dict(torch.load(pre_trained_net))
                model.eval()
                val_output = model(val_x)
                all_output.append(val_output)
            val_output = torch.cat(all_output,dim=1)
            val_output = inverse_10(val_output)
            val_y = inverse_10(val_y)

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
        plt.figure(figsize = (20,5))
        plt.plot(val_output.detach().cpu().numpy()[5],linestyle = '--' )
        plt.plot(val_y.detach().cpu().numpy()[5])
        plt.savefig('valid3.png', dpi=300)
    return losses.avg

def validate_single(val_loader,model,criterion,mode):
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
    for i in range(10,20):
        plt.plot(val_y.squeeze(1)[i].detach().cpu().numpy())
        plt.plot(val_output.detach().cpu().numpy()[i], linestyle='--')    
        plt.savefig(f'images_AdamW{i}.png', dpi=300)
        plt.close()
    return losses.avg
if __name__ == "__main__":
    main()