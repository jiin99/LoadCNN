import numpy as np
from matplotlib import pyplot as plt
import os
import random

path = "./log"
file_list = os.listdir(path)
print(file_list)

directory = ['real.npy','RNN1.npy','RNN3.npy', 'RNN5.npy','lstm1.npy','lstm3.npy','lstm5.npy','bi_lstm3.npy','bi_lstm5.npy', 'small_cnn.npy','big_cnn.npy',]
name = ['Real Usage','RNN/1','RNN/3','RNN/5','LSTM/1','LSTM/3','LSTM/5','Bi-LSTM/3','Bi-LSTM/5','smallCNN','bigCNN']
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
for i in directory : 
    ll = np.load('./log/' + i)
    print(len(ll))

for epoch in range(80):
    t = random.randrange(0,21015)
    print(t)
    plt.figure(figsize = (10,5))
    # plt.set_size_inches(18.5, 10.5)
    for idx,i in enumerate(directory) :
        log = np.load('./log/' + i)
        if i == 'real.npy':
            plt.plot(log[t:t+48],'black',label = 'Real Usage')
            plt.legend('real')
        else :
            # print(i,':',name[idx]) 
            plt.plot(log[t:t+48],linestyle = '--',label = name[idx])
        
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #   fancybox=True, shadow=True, ncol=11)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))#(loc='lower right',bbox_to_anchor = (1.2,0.3))
        plt.legend(loc = 'upper left')
        plt.ylim(0,5)
        plt.xlabel('time(30min)')
        plt.ylabel('electricity usage(kWh)')
    plt.savefig(f'./exp_fig/{epoch}.png', dpi=100)
    plt.close()




# aa = np.load('./log/RNN5.npy')
# # plt.figure(figsize = (10,5))
# print(len(aa)/48) #21015
# print(aa.shape) #1008720
# plt.plot(aa[:96],linestyle = '--' )
# plt.savefig('first.png', dpi=100)