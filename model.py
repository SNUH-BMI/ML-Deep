import torch
import torch.nn as nn
import torch.optim as optim

import tools as T 
import tools.torch

from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

############
### LSTM ###
############
class LSTM(nn.Module):

    def __init__(self, hidden_size=50, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, cell

    def forward(self, x):
        # hidden, cell state init
        h, c = self.init_hidden(x.size(0))
        h, c = h.to(x.device), c.to(x.device)

        out, (h, c) = self.lstm(x, (h, c))
        out = self.fc(out[:, -1])

        return out

############
#### GRU ###
############
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRU, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)   
        
        return out

############
## ResNet ##
############
class ReduceBlock1(nn.Module): # stride size =1
    def __init__(self, in_channel=1, out_channel=64, kernel_size=5,drop_rate=0.5):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Dropout(drop_rate)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        return self.layers(x)

class ReduceBlock2(nn.Module): # stride size =2
    def __init__(self, in_channel=1, out_channel=64, kernel_size=11,drop_rate=0.5):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Dropout(drop_rate)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        return self.layers(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, in_channel, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm1d(in_channel),
        nn.ReLU(),
        nn.Conv1d(in_channel, in_channel, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm1d(in_channel),
        ]
        self.layers = nn.Sequential(*layers)#

    def forward(self, x):
        return self.layers(x)

class CNN_Res(nn.Module):
    def __init__(self, in_channel=1, n_repeat=2):
        super().__init__()
        blocks=[ResBlock(in_channel=in_channel) for i in range(n_repeat)]

        self.blocks = nn.Sequential(*blocks)
        self.activation = nn.ReLU()

    def forward(self, x):
        for block in self.blocks:
            x = self.activation(block(x)+x)

        return x

class CNN7(nn.Module):
    '''Overall depth reduced version'''
    
    def __init__(self, n_classes=2, drop_rate=0.5):
        n_hidden_list = [512,256,64,16,n_classes]
        super().__init__()
        # 1st block = 12, 64, 11 # 12 leads일 때 ! lead 개수 바꾸면 in_Channel만 바꿔주면 됨. 
        layers=[
        ReduceBlock2(in_channel=1, out_channel=64, kernel_size=11,drop_rate=drop_rate),
        ReduceBlock2(in_channel=64, out_channel=64, kernel_size=7,drop_rate=drop_rate),
        ReduceBlock2(in_channel=64, out_channel=64, kernel_size=5,drop_rate=drop_rate),
        CNN_Res(in_channel=64, n_repeat=2),
        ReduceBlock1(in_channel=64, out_channel=128, kernel_size=5,drop_rate=drop_rate),
        CNN_Res(in_channel=128, n_repeat=2),
        ReduceBlock1(in_channel=128, out_channel=256, kernel_size=5,drop_rate=drop_rate),
        CNN_Res(in_channel=256, n_repeat=2),
        ReduceBlock1(in_channel=256, out_channel=512, kernel_size=5,drop_rate=drop_rate),
        CNN_Res(in_channel=512, n_repeat=2),
        ReduceBlock1(in_channel=512, out_channel=1024, kernel_size=5,drop_rate=drop_rate),
        CNN_Res(in_channel=1024, n_repeat=2),
        nn.Flatten(1),
        T.torch.model.FNN(n_hidden_list=n_hidden_list, activation_list=[nn.LeakyReLU()]*(len(n_hidden_list)-1)+[nn.Identity()])
        ] 
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #print(self.layers(x).shape)
        return self.layers(x)

############
## 2D CNN ##
############
class CNN_ecg_light(nn.Module):
        def __init__(self,drop_rate,SET_LAYER_NUM):
            super(CNN_ecg_light, self).__init__()        
            self.conv1 = nn.Sequential(
                nn.Conv2d(1,8, kernel_size=(1,10)), # kernal 8개, kernal shape 1x10 : 12x5000 -> 12x4991
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d((1,4)), # 12x4991 -> 6x1247 (DBP,SBP,HR은 3x1366이라 maxpool 2,4 못해)
                nn.Dropout(drop_rate)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=(1,10)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d((1,4)),
                nn.Dropout(drop_rate)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(16,32, kernel_size=(1,10)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d((3,4)),
                nn.Dropout(drop_rate)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(1,10)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d((1,4)),
                nn.Dropout(drop_rate)
            )
            self.hidden1 = nn.Linear(128, SET_LAYER_NUM) # 12x5000에서는 1024(=64x16)
            self.bn1 = nn.BatchNorm1d(SET_LAYER_NUM)
            self.elu = nn.ELU()
            self.hidden2 = nn.Linear(SET_LAYER_NUM, 10)
            self.bn2 = nn.BatchNorm1d(10)
            self.hidden3 = nn.Linear(10, 2)

            
        def forward(self, x):
            out = self.conv1(x)
            #print(out.shape)
            out = self.conv2(out)
            #print(out.shape)
            out = self.conv3(out)
            #print(out.shape)
            out = self.conv4(out)
            #print(out.shape)
            out = out.view(out.size(0), -1)
            #### normalize ECG waveform value & concatenate!!
            #out = manual_whole_normalize(out)
            out = self.elu(self.bn1(self.hidden1(out)))
            out = self.elu(self.bn2(self.hidden2(out)))
            #print(out.shape)
            out = self.hidden3(out)
            #print(out.shape)
            return F.softmax(out, dim=1)

############
## 1D CNN ##
############
class CNNClassifier(nn.Module):
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNNClassifier, self).__init__()
        
        self.dbp_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=2, dilation=2), #1*1418 --> 8 * 1 * (N-kernal_size/stride + 1 = 707)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, dilation=2), # 352
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, dilation=2), # 174
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, dilation=2), # 85
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(85),
        )
        self.sbp_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=2, dilation=2),#1*1424 --> 8 * 1 * (N- dilation*kernal_size/stride + 1 = 710)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, dilation=2), # 353
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, dilation=2), # 175
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, dilation=2), # 86
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(86),
        )
        self.hr_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=2, dilation=2),#1*1365 --> 8 * 1 * (N-kernal_size/stride + 1 = 681)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, dilation=2), # 319
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, dilation=2), # 158
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, dilation=2), # 77
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(77),
        )
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(1539, 512) # 230+226+170 
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(drop_rate)
        self.avgpool = nn.AvgPool1d(4)
        
        # gpu로 할당
        if use_cuda:
            self.dbp_layer = self.dbp_layer.to(device)
            self.sbp_layer = self.sbp_layer.to(device)
            self.hr_layer = self.hr_layer.to(device)
            self.relu = self.relu.to(device)
            self.sigmoid = self.sigmoid.to(device)
            self.fc1 = self.fc1.to(device)
            self.fc2 = self.fc2.to(device)
            self.fc3 = self.fc3.to(device)
            
            self.dropout = self.dropout.to(device)
        
    def forward(self, x, y, z, age_sex_gfr):
        x_out = self.dbp_layer(x)
        y_out = self.sbp_layer(y)
        z_out = self.hr_layer(z)
        out = torch.cat((x_out.squeeze(), y_out.squeeze(), z_out.squeeze(), age_sex_gfr))
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.sigmoid(out)
        
        return out

class CNN_ecg_light_1d(nn.Module):
    def __init__(self,drop_rate,SET_LAYER_NUM):
        super(CNN_ecg_light_1d, self).__init__()        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,16, kernel_size=7), # 맨 앞에 숫자는 채널을 의미함. 일단 eeg만 할때는 채널=1, 8은 아웃풋 개수
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # 12x4991 -> 6x1247 (DBP,SBP,HR은 3x1366이라 maxpool 2,4 못해)
            nn.Dropout(drop_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32,64, kernel_size=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop_rate)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop_rate)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop_rate)
        )
        #이거 숫자 알려면 forward의 view 이후 print
        self.hidden1 = nn.Linear(3072, SET_LAYER_NUM) #20,3520
        self.bn1 = nn.BatchNorm1d(SET_LAYER_NUM)
        self.elu = nn.ELU()
        self.hidden2 = nn.Linear(SET_LAYER_NUM, 10) 
        self.bn2 = nn.BatchNorm1d(10) #원래 10
        self.hidden3 = nn.Linear(10, 5) # (input dim, output dim) 따라서 5 의미는 multiclass 개수가 5개니까 (0,1,2,3,5)

        
    def forward(self, x):
        out = self.conv1(x)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = self.conv3(out)
        #print(out.shape)
        out = self.conv4(out)
        #print(out.shape)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)
        #print('forward view',out.shape)
        #### normalize ECG waveform value & concatenate!!
        out = self.elu(self.bn1(self.hidden1(out)))           
        out = self.elu(self.bn2(self.hidden2(out)))
        #print(out.shape)
        out = self.hidden3(out)
        #print(out.shape)
        return F.softmax(out, dim=1)