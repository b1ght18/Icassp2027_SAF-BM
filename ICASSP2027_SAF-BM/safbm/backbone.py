"""CNN14 with one BN branch per domain and a shared linear source head.

CNN14 architecture: PANNs, Qiuqiang Kong et al. (MIT; licenses/PANNs-MIT.txt).
This compact implementation preserves the local MCnn14 parameter names and
forward operations for checkpoint compatibility. The domain BN wrapper and
feature readout are used by SAF-BM's experiment modules.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_tasks):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bnF = nn.ModuleList(nn.BatchNorm2d(out_channels) for _ in range(nb_tasks))
        self.bnS = nn.ModuleList(nn.BatchNorm2d(out_channels) for _ in range(nb_tasks))

    def forward(self, x, pool_size=(2, 2), pool_type='avg', task=0):
        x = F.relu_(self.bnF[task](self.conv1(x)))
        x = F.relu_(self.bnS[task](self.conv2(x)))
        if pool_type == 'avg':
            return F.avg_pool2d(x, pool_size)
        if pool_type == 'max':
            return F.max_pool2d(x, pool_size)
        if pool_type == 'avg+max':
            return F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
        raise ValueError(f'Unknown pooling: {pool_type}')


class MCnn14(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320,
                 mel_bins=64, fmin=50, fmax=14000, classes_num=10, nb_tasks=3):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
            fmax=fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.ModuleList(nn.BatchNorm2d(mel_bins) for _ in range(nb_tasks))
        channels = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f'conv_block{i+1}', ConvBlock(channels[i], channels[i+1], nb_tasks))
        self.fc = nn.Linear(2048, classes_num)

    def features(self, waveforms, task=0):
        x = self.logmel_extractor(self.spectrogram_extractor(waveforms))
        x = self.bn0[task](x.transpose(1, 3)).transpose(1, 3)
        for i in range(1, 7):
            x = getattr(self, f'conv_block{i}')(x, task=task)
            x = F.dropout(x, p=0.2, training=self.training)
        x = x.mean(dim=3)
        return x.max(dim=2).values + x.mean(dim=2)

    def forward(self, waveforms, task=0):
        return self.fc(self.features(waveforms, task))

    def get_output_dim(self):
        return self.fc.out_features

    def freeze_weight(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
