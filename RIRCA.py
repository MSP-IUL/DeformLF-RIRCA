import torch
import torch.nn as nn

# Default convolution layer with kernel size padding
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=7):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Residual Channel Attention Block with Projection
class RCABP(nn.Module):
    def __init__(self, conv, in_feat, n_feat, kernel_size, reduction, bias=False, bn=True, act=nn.LeakyReLU(0.1)):
        super(RCABP, self).__init__()
        modules_body = [conv(in_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias), CALayer(n_feat, reduction)]
        self.body = nn.Sequential(*modules_body)
        self.proj = conv(in_feat, n_feat, 1, bias=bias)
        

    def forward(self, x):
        res = self.body(x)
        res += self.proj(x)
        return res

# Residual Group without Self-Attention
class ResidualGroup(nn.Module):
    def __init__(self, conv, in_feat, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [RCABP(conv, in_feat, n_feat, kernel_size, reduction)]
        for _ in range(n_resblocks - 1):
            modules_body.append(RCABP(conv, n_feat, n_feat, kernel_size, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return torch.cat([x, res], 1)

# DenseNet with Residual Channel Attention blocks (DRCA)
class DRCA(nn.Module):
    def __init__(self, conv=default_conv):
        super(DRCA, self).__init__()
        
        # Model parameters - set as constants
        n_resgroups = 4       # Example value
        n_resblocks =5    # Example value
        n_feats = 49          # Matches the view dimension
        kernel_size = 3
        reduction = 7
        act = nn.LeakyReLU(0.1)

        # Define head module to accept 49 channels directly
        modules_head = [conv(49, n_feats, kernel_size)]

        # Define body module with residual groups
        modules_body = [ResidualGroup(conv, (i + 1) * n_feats, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks) for i in range(n_resgroups)]
        modules_body.append(conv((n_resgroups + 1) * n_feats, n_feats, 1))

        # Define tail module to output the same number of channels as input (49)
        modules_tail = [conv(n_feats, 49, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x  # Residual connection to maintain input shape
        x = self.tail(res)
        return x
