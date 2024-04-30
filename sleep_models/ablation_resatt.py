# Modified from https://github.com/DiZhang-XDU/sleepstager-twobranch/blob/main/sleep_models/ResAtt_OneLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelMask(nn.Module):
    def __init__(self, input_channels, ratio=8):
        super(ChannelMask, self).__init__()
        squeeze_size = int(input_channels / ratio)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(input_channels, squeeze_size, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(squeeze_size, input_channels, 1)
        self.mlp = nn.Sequential(self.fc1, self.relu, self.fc2)
    def forward(self, x):
        x1 = self.avg_pooling(x)
        x1 = self.mlp(x1)
        x2 = self.max_pooling(x)
        x2 = self.mlp(x2)
        return torch.sigmoid(x1 + x2)

class TemporalMask(nn.Module):
    def __init__(self, kernel_size = 7):
        super(TemporalMask, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size = kernel_size, padding = padding)
    
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.sigmoid(max_out) + 1
        return out

class BlockV2(nn.Module):
    expansion = 1
    def __init__(self, input_channels, output_channels, stride = 1, downsample = None):
        super(BlockV2, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, 7, stride, padding=3, bias = False)
        self.bn1 = nn.BatchNorm1d(output_channels)

        self.conv2 = nn.Conv1d(output_channels, output_channels, 7, stride = 1, padding=3, bias = False)
        self.bn2 = nn.BatchNorm1d(output_channels)

        self.chn = ChannelMask(output_channels)
        self.seq = TemporalMask()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        cm = self.chn(out)
        out = cm * out
        tm = self.seq(out)
        out = tm * out

        out += residual 

        return out

class Backbone(nn.Module):
    def __init__(self, input_channels, layers=[1, 1, 1, 1], num_classes=6):
        self.inplanes = 64
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BlockV2, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(BlockV2, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BlockV2, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(BlockV2, 512, layers[3], stride=2)


    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x


class Score_Stage_150s(nn.Module):
    def __init__(self, input_channels = 5, len_ep = 3840):
        super(Score_Stage_150s, self).__init__()
        self.input_channels = input_channels
        self.len_ep = len_ep
        self.net = Backbone(input_channels)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.bn1 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc2_1 = nn.Conv1d(256, 48, kernel_size=1)
    

    def forward(self, x:torch.Tensor):
        B, C, L = x.shape
        x = x.swapaxes(1,2).unfold(1, self.len_ep//6, self.len_ep//6).reshape([-1, C, self.len_ep//6])   # 30s / 5s 
        x = self.net(x)
    
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.gap(x).squeeze(-1)
        x = x.reshape([B, -1, x.size(-1)])  # B, C, L
        x = x.swapaxes(1,2)                 # B, L, C
        x = self.fc2_1(x)
        x = x.swapaxes(1,2)                 # B, C, L
        return x, None



if __name__ == "__main__":
    pass
