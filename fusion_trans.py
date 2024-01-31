import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import MPCNN_model
import config
from model import Transformer
from spectrogram_trans_model.spectrogram_util import trans_to_spectrogram

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#归一化到【-1，1】
def normalization(data):
    _range = np.max(abs(data))
    return data / _range
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(194)
        self.conv1 = Conv(32, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm1d(194)
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm1d(194)
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(32, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv1d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm1d(64)
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()
        self.fc1 = nn.Conv1d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.compress = ChannelPool()
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
        self.relu = nn.ReLU(inplace=True)
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout(drop_rate)
        self.drop_rate = drop_rate
    def forward(self, g, x):
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)
        g = self.compress(g)
        g = self.sigmoid(g)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        bp = bp.reshape(32, -1)
        fuse = self.residual(torch.cat([g, x, bp], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))
class Up(nn.Module):
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch)
        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None
    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)
class TransFuse_S(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_S, self).__init__()
        self.cnn = MPCNN_model.mpcnn_model().to(device)
        self.transformer = Transformer(device='cuda:0', d_feature=config.SIG_LEN, d_model=config.d_model, d_inner=config.d_inner,n_layers=config.num_layers, n_head=config.num_heads, d_k=64, d_v=64, dropout=0.1, class_num=5)
        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)
        self.final_x = nn.Sequential(
            Conv(32, 16, 1, bn=True, relu=True),
            Conv(16, 8, 3, bn=True, relu=True),
            Conv(8, 5, 3, bn=False, relu=False)
            )
        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.up_c = BiFusion_block(ch_1=32, ch_2=32, r_2=4, ch_int=64, ch_out=32, drop_rate=drop_rate/2)
        self.drop = nn.Dropout2d(drop_rate)
        self.fc = nn.Linear(194,5)
        self.fc0 = nn.Linear(5226,400)
    def forward(self, input, labels=None,RR=None):
        input1 = np.abs(librosa.stft(input.cpu().numpy(), n_fft=400, win_length=64))
        input1 = librosa.amplitude_to_db(input1, ref=np.max)
        input1 = input1.reshape(32, -1)
        input1 = normalization(input1)
        input1 = torch.tensor(input1).float()
        input1 = input1.to(device)
        input1 = self.fc0(input1)
        x_b = self.transformer(input1, RR)
        x_b = self.drop(x_b)
        x_u = self.cnn(input, RR)
        x_u = self.drop(x_u)
        x_c = self.up_c(x_u, x_b)
        map_x = self.fc(x_c)
        return map_x
