import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import nn
from torch.nn import functional as F
import torch.nn
from torchsummary import summary

#GCT注意力机制
from Attention.area.non_local import _NonLocalBlockND, NonLocalBlock
from Attention.channel.ECAAttention import ECAAttention
from Attention.mix.CBAM import CBAMBlock


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((-3, 2), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / \
                (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate
class mpcnn_model(nn.Module):
    def __init__(self):
        super(mpcnn_model, self).__init__()
        self.ECA = ECAAttention(kernel_size=3)
        #M11
        self.con_m11 = nn.Conv1d(1, 32, 20, 1, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.maxpool_m11_1 = nn.MaxPool1d(3)
        self.maxpool_m11_2 = nn.MaxPool1d(5)
        self.maxpool_m11_3 = nn.MaxPool1d(7)
        #M21
        self.con_m21 = nn.Conv1d(1, 32, 25, 1, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.maxpool_m21_1 = nn.MaxPool1d(3)
        self.maxpool_m21_2 = nn.MaxPool1d(5)
        self.maxpool_m21_3 = nn.MaxPool1d(7)
        # M31
        self.con_m31 = nn.Conv1d(1, 64, 25, 1, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.maxpool_m31_1 = nn.MaxPool1d(7)
        # M32
        self.con_m32 = nn.Conv1d(64, 32, 15, 1, 1)
        self.bn4 = nn.BatchNorm1d(32)
        self.maxpool_m32_1 = nn.MaxPool1d(3)
        # M41
        self.con_m41 = nn.Conv1d(1, 32, 30, 1, 1)
        self.bn5 = nn.BatchNorm1d(32)
        self.maxpool_m41_1 = nn.MaxPool1d(4)
        self.maxpool_m41_2 = nn.MaxPool1d(6)
        self.maxpool_m41_3 = nn.MaxPool1d(8)
        # M51
        self.con_m51 = nn.Conv1d(1, 64, 30, 1, 1)
        self.bn6 = nn.BatchNorm1d(64)
        self.maxpool_m51_1 = nn.MaxPool1d(7)
        # M52
        self.con_m52 = nn.Conv1d(64, 32, 15, 1, 1)
        self.bn7 = nn.BatchNorm1d(32)
        self.maxpool_m52_1 = nn.MaxPool1d(3)
        self.fc1 = nn.Linear(1092, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 86)
        self.fc4 = nn.Linear(86, 64)
    def forward(self, input, RR):
        m11_out = self.con_m11(input)
        m11_out = self.bn1(m11_out)
        m11_out = F.relu(m11_out)
        m11_out = self.maxpool_m11_1(m11_out)
        m11_out = self.maxpool_m11_2(m11_out)
        m11_out = self.maxpool_m11_3(m11_out)
        m21_out = self.con_m21(input)
        m21_out = self.bn2(m21_out)
        m21_out = F.relu(m21_out)
        m21_out = self.maxpool_m21_1(m21_out)
        m21_out = self.maxpool_m21_2(m21_out)
        m21_out = self.maxpool_m21_3(m21_out)
        m31_out = self.con_m31(input)
        m31_out = self.bn3(m31_out)
        m31_out = F.relu(m31_out)
        m31_out = self.maxpool_m31_1(m31_out)
        m32_out = self.con_m32(m31_out)
        m32_out = self.bn4(m32_out)
        m32_out = F.relu(m32_out)
        m32_out = self.maxpool_m32_1(m32_out)
        m41_out = self.con_m41(input)
        m41_out = self.bn5(m41_out)
        m41_out = F.relu(m41_out)
        m41_out = self.maxpool_m41_1(m41_out)
        m41_out = self.maxpool_m41_2(m41_out)
        m41_out = self.maxpool_m41_3(m41_out)
        m51_out = self.con_m51(input)
        m51_out = self.bn6(m51_out)
        m51_out = F.relu(m51_out)
        m51_out = self.maxpool_m31_1(m51_out)
        m52_out = self.con_m52(m51_out)
        m52_out = self.bn7(m52_out)
        m52_out = F.relu(m52_out)
        m52_out = self.maxpool_m52_1(m52_out)
        m11_out = m11_out.reshape(32, -1, 2)
        m21_out = m21_out.reshape(32, -1, 2)
        m41_out = m41_out.reshape(32, -1, 2)
        m32_out = m32_out.reshape(32, -1, 2)
        m52_out = m52_out.reshape(32, -1, 2)
        output = torch.cat((m11_out,m21_out,m32_out,m41_out,m52_out), dim=1)
        output = output.view(output.size(0), -1)
        output = torch.cat([output, RR], 1)
        output = output.to(torch.float32)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = self.fc4(output)
        return output

