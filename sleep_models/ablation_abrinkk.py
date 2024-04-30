# Modified from https://github.com/abrinkk/psg-age-estimation/blob/main/m_psg2label.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# from base.base_model import BaseModel
# from config import Config


class M_PSG2FEAT(nn.Module):
    def __init__(self, config = 'never mind'):
        """A model to process epochs of polysomnography data

        Args:
            config: An instance of the config class with set attributes.
        """
        super().__init__()
        # Attributes
        self.n_channels = 6 #config.SWIN.IN_CHANS
        self.n_class = 48 #config.SWIN.EMBED_DIM
        self.n_label = None #len(config.pre_label)
        self.return_only_pred = True #config.return_only_pred
        self.return_pdf_shape = None #config.return_pdf_shape
        
        ### LAYERS ###
        self.channel_mixer = nn.Sequential(
                nn.Conv2d(1, 32, (self.n_channels, 1), bias = False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True))
        # self.channel_mixer = nn.Sequential(
        #         nn.Conv2d(self.n_channels, 32, (1, 1), bias = False),
        #         nn.BatchNorm2d(32),
        #         nn.ReLU6(inplace=True))
        self.MobileNetV2 = MobileNetV2(num_classes = self.n_class)
        self.LSTM = nn.LSTM(128, 48//2, num_layers = 1, bidirectional = True)
        # self.add_attention = AdditiveAttention(256, 512)
        # self.linear_l = nn.Linear(256, 256)
        # self.dropout = nn.Dropout(p = .75)#config.SWIN.DROP_RATE)
        # Label specific layers
        # self.classify_bias_init = [50.0]#, 10.0]
        # self.classify_l = nn.Linear(256, self.n_class)
        # self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)
        
    def forward(self, X):
        """Forward call of model

        Args:
            X (Tensor): Input polysomnography epoch of size [batch_size, n_channels, 38400]

        Returns:
            dict: A dict {'pred': age predictions, 
                          'feat': latent space representation,
                          'alpha': additive attention weights,
                          'pdf_shape': shape of predicted age distribution (not used)}
        """
        # X.size() = [Batch_size, Channels = 13, Time = 5*60*128]
        X = torch.unsqueeze(X, 1)
        # X.size() = [Batch_size, Feature_maps = 1, Channels = 13, Time = 5*60*128]
        # Channel Mixer
        X = self.channel_mixer(X)
        # X.size() = [Batch_size, Feature_maps = 32, Channels = 1, Time = 5*60*128]
        # Modified MobileNetV2
        X = self.MobileNetV2(X)
        # X.size() = [Batch_size, Feature_maps = 320, Channels = 1, Time = 5*60*16]
        # LSTM layer
        X = X.view(-1, X.size(1), 1, int(X.size(3) / (10*4)), 10*4)
        X = torch.squeeze(X.mean([4]), 2)
        X = X.permute(2, 0, 1)
        self.LSTM.flatten_parameters()
        X, _ = self.LSTM(X)
        # # Attention layer
        X = X.permute(1, 0, 2)
        return X, None  # B, 30, 48


        # # Averaged features
        # X_avg = torch.mean(X, 1)
        # X_a, alpha = self.add_attention(X)
        # # Linear Transform
        # X_a = self.linear_l(F.relu(X_a))
        # # Dropout
        # X_a = self.dropout(X_a)
        # # Linear
        # C = self.classify_l(X_a)

        # C = torch.squeeze(C, 1)
        # if self.return_only_pred:
        #     return torch.unsqueeze(C[:, 0], 1)
        # else:
        #     return {'pred': C[:, 0], 'feat': torch.cat((X_a, X_avg), 1), 'alpha': alpha, 'pdf_shape': C[:, 1]}


# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, (1,2)],
                [6, 32, 2, (1,2)],
                [6, 64, 2, (1,2)],
                [6, 128, 1, (1,1)],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(input_channel, input_channel, stride=(1,2))]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else (1,1)
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
#        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=(1,1)))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
#        self.classifier = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(self.last_channel, num_classes),
#        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
#        x = x.mean([2, 3])
#        x = self.classifier(x)
        return x

# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=(1,3), stride=(1,1), groups=1):
        padding = tuple((np.array(kernel_size) - 1) // 2)
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

# Dilated conv block (conv->bn->relu->maxpool->dropout)
class dilated_conv_block(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(1,3), stride=(1,1), dilation=(1,1), groups=1, max_pool=(1,1), drop_chance=0.0):
        padding = tuple(((np.array(kernel_size)-1)*(np.array(dilation)-1) + np.array(kernel_size) - np.array(stride)) // 2)
        if max_pool[1] > 1:
            super(dilated_conv_block, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation , groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pool, stride=max_pool),
                nn.Dropout(p=drop_chance)
            )
        else:
            super(dilated_conv_block, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation , groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_chance)
            )

# Additive Attention
class AdditiveAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        """Additive attention module

        Args:
            input_size (int): Size of input
            hidden_size (int): Size of hidden layer
        """
        super(AdditiveAttention, self).__init__()
        self.linear_u = nn.Linear(input_size, hidden_size)
        self.linear_a = nn.Linear(hidden_size, 1, bias = False)
        
    def forward(self, h):
        """Forward call of model

        Args:
            h (Tensor): Input features of size [batch_size, ..., input_size]

        Returns:
            s (Tensor): Summary features
            a (Tensor): Additive attention weights
        """
        # h.size() = [Batch size, Sequence length, Hidden size]
        u = torch.tanh(self.linear_u(h))
        a = F.softmax(self.linear_a(u), dim = 1)
        s = torch.sum(a * h, 1)
        return s, a
        
# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride[1] in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride[1] == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=(1,1)))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, (1,1), (1,1), 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


if __name__ == "__main__":
    pass

    net = M_PSG2FEAT()
    x = torch.rand([16, 6, 19200])
    print(net(x).shape)