import numpy as np

import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    # 利用 to 转换
    device = torch.device('cuda')
dtype = torch.float
filename = 'flights.xlsx'
filenametest = 'flights.xlsx'
full_path = 'D:\\dataset\\{}'.format(filename)
full_pathtest = 'D:\\dataset\\{}'.format(filenametest)

torch.random.seed()
np.random.seed()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：
# __len()__, __getitem()__
class FlightsDataSet(Dataset):
    def __init__(self, file_dir, name, N, device=True):
        self.file_dir = file_dir
        self.time_value = []
        self.size = 0
        self.N = N;
        self.device=device
        # self.transform = transform
        df = pd.read_excel(self.file_dir)  # 这个会直接默认读取到这个Excel的第一个表单
        self.time_value = list(df[name])
        self.size = len(self.time_value) - self.N  # 个数 非索引 索引减一
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.time_value = scaler.fit_transform(np.array(self.time_value).reshape(-1, 1))  # 归一化，转为二维tensor
    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        value = np.array(self.time_value[idx: idx + self.N])
        label = np.array(self.time_value[idx + self.N])

        if self.device:
            device = torch.device('cuda')
            value=torch.tensor(value,device=device,dtype=torch.float)
            label=torch.tensor(label,device=device,dtype=torch.float)

        sample = {'value': value, 'label': label}
        return sample


train_dataset = FlightsDataSet(full_path, "price", 12)

trainset_dataloader = DataLoader(dataset=train_dataset,
                                 batch_size=1,
                                 shuffle=False)

# for i_batch, sample_batch in enumerate(trainset_dataloader):
#     value_batch, labels_batch = \
#         sample_batch['value'], sample_batch['label']
#     if i_batch==3:
#         value_batch=value_batch.view(12)
#         print(value_batch.shape)
#         print(labels_batch.size())
#
# a=torch.tensor([[1,2,3,4,5,6,7,8]])
# print(a.shape)
# a=a.view(2,4)
# print(a.shape)
#####################

#
#
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM().to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)
epochs = 100

for i in range(epochs):
    for i_batch, sample_batch in enumerate(trainset_dataloader):
        seq, labels = \
            sample_batch['value'], sample_batch['label']
        seq=seq.view(12)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size,device=device),
                             torch.zeros(1, 1, model.hidden_layer_size,device=device))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

