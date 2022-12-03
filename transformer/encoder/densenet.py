import torch
import torch.nn as nn
import math

# Dense layer with bottleneck
class _DenseLayer(nn.Module):
    def __init__(self, num_features_in, growth_rate, dropout):
        super(_DenseLayer, self).__init__()
        # BN-relu-1x1conv-BN-relu-3x3conv
        # For bottleneck, we let 1x1 conv produce 4*growth_rate feature-maps
        self.norm1 = nn.BatchNorm2d(num_features_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_features_in, 4 * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # bottleneck
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        out = self.dropout(out)
        return out

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_features_in, growth_rate, dropout):
        super(_DenseBlock, self).__init__()

        self.layers = nn.ModuleList(
            [
                _DenseLayer(num_features_in + i * growth_rate, growth_rate, dropout)
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        x [batch_size, num_features, height, width]
        out [batch_size, num_features + block_depth*growth_rate, height, width]
        """
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1)) # concatenate channels (index 1)
            features.append(new_feature)
        return torch.cat(features,1)

class _Transition(nn.Module):
    def __init__(self, num_features_in, num_features_out):
        # BN-1x1conv-2x2avgpool
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_features_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_features_in, num_features_out, kernel_size=1, stride=1, bias = False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        x [batch_size, channels, height, width]
        out [batch_size, channels*compression, height/2, width/2]
        """
        out = self.relu(self.norm(x))
        out = self.conv(out)
        return self.pool(out)
        

class DenseNet(nn.Module):
    """
    DenseNet-B consisting of three dense blocks
    """
    def __init__(self, growth_rate=24, block_depth=16, compression=0.5, dropout=0.2):
        super(DenseNet, self).__init__()
        num_features = 2 * growth_rate
        # Initial convolution + pooling
        self.conv0 = nn.Conv2d(1, num_features, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm0 = nn.BatchNorm2d(num_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # DenseB-Transition-DenseB-Transition-DenseB
        self.block1 = _DenseBlock(block_depth, num_features, growth_rate, dropout)
        num_features = num_features + block_depth*growth_rate
        self.trans1 = _Transition(num_features, math.floor(num_features * compression))
        num_features = math.floor(num_features * compression)

        self.block2 = _DenseBlock(block_depth, num_features, growth_rate, dropout)
        num_features = num_features + block_depth*growth_rate
        self.trans2 = _Transition(num_features, math.floor(num_features * compression))
        num_features = math.floor(num_features * compression)

        self.block3 = _DenseBlock(block_depth, num_features, growth_rate, dropout)
        num_features = num_features + block_depth*growth_rate

        # Final batch norm
        self.post_norm = nn.BatchNorm2d(num_features)
        self.out_features = num_features

    def forward(self, x):
        """
        x [batch_size, channels=1, height, width]
        out [batch_size, self.out_features, height/16, width/16]
        """
        batch_size, _, height, width = x.size()
        # Initial convolution + pooling
        # hw/4
        out = self.pool0(self.relu0(self.norm0(self.conv0(x))))

        # DenseB-Transition-DenseB-Transition-DenseB
        out = self.block1(out)
        # hw/2
        out = self.trans1(out)
        out = self.block2(out)
        # hw/2
        out = self.trans2(out)
        out = self.block3(out)
        out = self.post_norm(out)

        # assert (out.size() == (batch_size, self.out_features, round(height/16), round(width/16))), "CNN output incorrect shape"

        return out

