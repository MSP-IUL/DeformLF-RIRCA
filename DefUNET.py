import torch
import torch.nn as nn
import torch.nn.functional as F
from Deformable import DeformableConv2d
from opt import opt  # Import the opt object with channel_size defined

class DeformableResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeformableResidualBlock, self).__init__()
        
        # Ensure channels are integers in case of ambiguity
        in_channels = int(in_channels) if isinstance(in_channels, torch.Tensor) else in_channels
        out_channels = int(out_channels) if isinstance(out_channels, torch.Tensor) else out_channels
        
        # Deformable convolution layers with LeakyReLU activation
        self.conv1 = DeformableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual connection with a possible adjustment for matching channels
        self.residual = nn.Identity() if in_channels == out_channels else DeformableConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = self.conv2(x)
        
        # Match dimensions by center-cropping if necessary
        if x.shape[2:] != residual.shape[2:]:
            diff_h = (x.size(2) - residual.size(2)) // 2
            diff_w = (x.size(3) - residual.size(3)) // 2
            x = x[:, :, diff_h: x.size(2) - diff_h, diff_w: x.size(3) - diff_w]

        x += residual
        return F.leaky_relu(x, negative_slope=0.01)

class DeformableEncoderDecoder(nn.Module):
    def __init__(self, opt):
        super(DeformableEncoderDecoder, self).__init__()
        in_channels = opt.channel_size
        out_channels = opt.channel_size
        
        # Encoder blocks without downsampling to maintain spatial dimensions
        self.enc1 = DeformableResidualBlock(in_channels, 64)
        self.enc2 = DeformableResidualBlock(64, 64)
        
        # Bottleneck block
        self.bottleneck = DeformableResidualBlock(64, 64)
        
        # Decoder blocks
        self.dec2 = DeformableResidualBlock(64, 64)
        self.dec1 = DeformableResidualBlock(64, 64)
        
        # Final output layer to match the in_channels and out_channels
        self.final_conv = DeformableConv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encoder path
        x = self.enc1(x)
        skip1 = x
        x = self.enc2(x)
        skip2 = x
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        x = self.dec2(x + skip2)
        x = self.dec1(x + skip1)
        
        # Final output
        x = self.final_conv(x)
        return x

# # Model initialization and testing
# if __name__ == "__main__":
#     model = DeformableEncoderDecoder(opt)
    
#     # Dummy input tensor with shape (49, 64, 96, 96)
#     x = torch.randn(49, opt.channel_size, 96, 96)
    
#     # Run the model
#     output = model(x)
    
#     # Output shape verification
#     print('Output Shape:', output.shape)  # Expected shape: (49, 64, 96, 96)
