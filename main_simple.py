#from cnn import CNN_3days,CNN_7days,CNN_BN_3days,CNN_BN_7days,CNN_GN_3days,CNN_GN_7days,CNN_3days_half,CNN_7days_half
#from rnn import RNN,LSTM
import cnn_simple
import dataloader_simple
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
import rnn
import argparse
import pandas as pd
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import wandb
wandb.init(project = 'paper_exp')
is_best = 0.65

parser = argparse.ArgumentParser(description='PyTorch code: ODIN')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for data loader')
parser.add_argument('--save_root', default = './ensemble',required=True, help='log save path')
parser.add_argument('--loss', type=str, default='rmse', help='criterion')
# parser.add_argument('--network', type=str, default='dense_C',  required=True, help='dense | res | vgg16')
parser.add_argument('--gpu', type=str, default='7', help='gpu index')
parser.add_argument('--days',type = str, default = '7', help ='data')
parser.add_argument('--tries',type = str, default = '1', required = True, help ='try')

args = parser.parse_args()

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self,yhat,y):
        mse_value = self.mse(yhat,y).mean(axis=1) + 1e-8
        return torch.sqrt(mse_value).mean()


def inverse_10(x):
  return x * 10

def main():
    global is_best

    if not os.path.exists(os.path.join(args.save_root,args.days +'days'+args.tries)):
        os.makedirs(os.path.join(args.save_root,args.days +'days'+args.tries))

    val_loss_history = []
    tr_loss_history = []
    test_loss_history = []
    train_logger = util.Logger(os.path.join(args.save_root,args.days +'days'+args.tries) + '/train.log')
    valid_logger = util.Logger(os.path.join(args.save_root,args.days +'days'+args.tries) + '/valid.log')
    test_logger = util.Logger(os.path.join(args.save_root,args.days +'days'+args.tries) + '/test.log')

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    model = cnn_simple.CNN_big_diff(int(args.days)).cuda()
    # model = rnn.LSTM(1, 32, 48, 336, 3, bidirectional=False).cuda()
    # model = rnn.RNN(1, 32, 48, 336, 5).cuda()
    # optimizer = optim.SGD(model.parameters(),lr=0.1, momentum=0.9, nesterov=True)#, weight_decay = 0.0001)
    optimizer = torch.optim.Adam(model.parameters())
# python main_simple.py --save_root ./bi_lstm_3_adam_real2 --days 7 --tries 1 --gpu 1
    # wandb.watch(model)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,25], gamma=0.1)
    #Set criterion
    if args.loss == 'rmse' : 
        criterion = RMSELoss().cuda()
    elif args.loss == 'mae' : 
        criterion =nn.L1Loss().cuda()
    
    train_loader, test_loader, valid_loader = dataloader_simple.data_set_base(int(args.days), args.batch_size)

    #Start Train
    for epoch in tqdm(range(0,60)):
        train(criterion,model,train_loader,optimizer,epoch,train_logger,tr_loss_history)
        prec = validate(valid_loader,model,criterion,epoch,'valid',valid_logger,val_loss_history)

        # if prec < is_best:
        #     if not os.path.isdir(os.path.join(args.save_root,args.days +'days'+args.tries) + '/best'):
        #         os.mkdir(os.path.join(args.save_root,args.days +'days'+args.tries) + '/best')
        #     torch.save(model.state_dict(),os.path.join(args.save_root,args.days +'days'+args.tries) + '/best' +'/ckpt_best{0}.pt'.format(epoch))
            # is_best = prec
        torch.save(model.state_dict(),os.path.join(args.save_root,args.days +'days'+args.tries) +'/ckpt_epoch{0}.pt'.format(epoch))
        # scheduler.step()
    torch.save(model.state_dict(),os.path.join(args.save_root,args.days +'days'+args.tries) +'/ckpt_last.pt')
    test_prec = validate(test_loader,model,criterion,epoch,'test',test_logger,test_loss_history)
    print('rmse test prec : ',test_prec)


def train(loss_fn,model,trn_dl,optimizer,epoch,logger,li):
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    model.train()
    end = time.time()
    tr_loss = 0
    for step,(x,y) in enumerate(trn_dl):
        
        data_time.update(time.time() - end)
        # x, y= x.unsqueeze(1).float().cuda(), y.squeeze().float().cuda()
        x = x.float().cuda()
        y = y.squeeze().float().cuda()
        output = model(x)
        # output = model(x)
        # import pdb; pdb.set_trace()
        loss = loss_fn(output,y)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        tr_loss += loss.item()
        losses.update(loss.item(), x.size(0))

        wandb.log({'step': step, 'train_loss_step': loss.item()})
        #wandb.log({'epoch': epoch, 'train_loss_avg': losses.avg})
        #wandb.log({'epoch': epoch, 'trtrtr_loss': tr_loss/(step+1)})
        if step % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, step, len(trn_dl), batch_time=batch_time,
                data_time=data_time, loss=losses))
            #wandb.log({'epoch': epoch, 'tr_loss': tr_loss/(step+1)})
            li.append(tr_loss/(step+1))
    wandb.log({'epoch': epoch, 'train_avg': tr_loss/(step+1)})
    logger.write([epoch, losses.avg])


def validate(val_loader,model,criterion,epoch,mode,logger,li):
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
            # val_x = val_x.float().cuda()
            # val_y = val_y.squeeze().float().cuda()
            # val_output = model(val_x)

            val_output = inverse_10(val_output)
            val_y = inverse_10(val_y)

            v_loss = criterion(val_output,val_y)
            losses.update(v_loss.item(), val_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            val_loss += v_loss.item()
            #if mode =='valid':
            #    wandb.log({'epoch': epoch, 'v_loss_avg': val_loss/(i+1)})
            #else :
            #    wandb.log({'epoch': epoch, 'v_loss_avg': val_loss/(i+1)})
            if i % 10 == 0:
                print(mode, ': [{0}/{1}]\t'
                      'V_Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'V_Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                                i, len(val_loader), batch_time=batch_time, loss=losses))
                #if mode =='valid':
                #    wandb.log({'epoch': epoch, 'valval_loss': losses.val})
                #    wandb.log({'epoch': epoch, 'valval_loss_avg': losses.avg})
                #    wandb.log({'epoch': epoch, 'vv_loss_avg': val_loss/(i+1)})
                #    li.append(val_loss/(i+1))
                #else :
                #    wandb.log({'epoch': epoch, 'testtest_loss': losses.val})
                #    wandb.log({'epoch': epoch, 'testtest_loss_avg': losses.avg})
                #    wandb.log({'epoch': epoch, 'vv_loss_avg': val_loss/(i+1)})
                #    li.append(val_loss/(i+1))
        if mode == 'valid':
            plt.figure(figsize = (20,5))
            plt.plot(val_output.detach().cpu().numpy()[5],linestyle = '--' )
            plt.plot(val_y.detach().cpu().numpy()[5])
            plt.savefig('valid3.png', dpi=300)
        else :
            plt.figure(figsize = (20,5))
            plt.plot(val_output.detach().cpu().numpy()[5],linestyle = '--' )
            plt.plot(val_y.detach().cpu().numpy()[5])
            plt.savefig('test3.png', dpi=300)

        wandb.log({'epoch': epoch, 'valid_avg': val_loss/(i+1)})
        logger.write([epoch, losses.avg])
    return losses.avg
if __name__ == "__main__":
    main()
