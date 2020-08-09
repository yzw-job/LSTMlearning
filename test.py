# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dtype = torch.float
device = torch.device("cuda")

# C:\Users\Administrator\seaborn-data
flight_data = sns.load_dataset('flights')

all_data = np.array(flight_data['passengers'].values.astype(float))
all_data=torch.from_numpy(all_data)

# all_data = torch.tensor(all_data, device=device, dtype=dtype)
# print(all_data.shape)
# print(all_data)
# plt.plot(flight_data['passengers'])
# plt.plot(a)
# plt.show()

# slpit test data
test_data_size = 12

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
print(len(train_data))
print(len(test_data))

# normalize
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))  # 归一化，转为二维tensor
# print(train_data_normalized.shape)  # (132,1)
# print(train_data_normalized[:5])

# convert to tensor
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)  # 转为一维
# print(train_data_normalized.shape)
#  there are 12 months in a year, Therefore, we will set the input sequence length for training to 12.
train_window = 12


# 训练数据为12个月的序列，正确解为第 12 + 1 个月的值


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))  # 一个个成对的元组
    return inout_seq


train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
for seq, labels in train_inout_seq:
    seq.view(len(seq), 1, -1)
    print(seq.shape)
    print(labels.size())
#
# # print(train_inout_seq[:5])
#
# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#
#         self.lstm = nn.LSTM(input_size, hidden_layer_size)
#
#         self.linear = nn.Linear(hidden_layer_size, output_size)
#
#         self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
#                             torch.zeros(1, 1, self.hidden_layer_size))
#
#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]
#
#
# model = LSTM()
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# print(model)
# epochs = 150
#
# for i in range(epochs):
#     for seq, labels in train_inout_seq:
#         optimizer.zero_grad()
#         model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
#                              torch.zeros(1, 1, model.hidden_layer_size))
#
#         y_pred = model(seq)
#
#         single_loss = loss_function(y_pred, labels)
#         single_loss.backward()
#         optimizer.step()
#
#     if i % 25 == 1:
#         print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
#
# print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
