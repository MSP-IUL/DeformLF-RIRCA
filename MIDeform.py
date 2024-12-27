import torch
import torch.nn as nn
import math
from opt import opt
from Deformable import DeformableConv2d
from Dataset import DatasetFromHdf5
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
class MI(nn.Module):
    def __init__(self, opt):
        super(MI, self).__init__()
        an2=opt.angular_out * opt.angular_out
        self.MIDeform = nn.Sequential(
            DeformableConv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            DeformableConv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            DeformableConv2d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.1),
            DeformableConv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            
        )
        self.MIDeform2 = nn.Sequential(
            DeformableConv2d(in_channels=an2, out_channels=an2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
           
           
            )
            
    def forward(self, LFI,img_source, opt):
        device = LFI.device  # To ensure all tensors are on the same device
        N, num_source, h, w = img_source.shape
        an = opt.angular_out
       
        # Reshape LFI to match the MIDeform input requirements
        LFI = LFI.view(-1, 1, 2,2)
        #rint('shape of LFI:',LFI.shape)

        # Pass through MIDeform layers
        LFI = self.MIDeform(LFI)
       

        # Pixel shuffle for upsampling
        pixel_shuffle = nn.PixelShuffle(4)
        LFI = pixel_shuffle(LFI)
      

        # Reshape to match the expected lens_data structure
        lens_data = LFI.view(1, h, w, 8, 8).to(device)
      

        # Lens2SAI transformation
        img = []
        for v in range(an):
            for u in range(an):
                img.append(lens_data[:, :, :, v, u].clone())

        img = torch.stack(img).to(device)
        img=img.permute(1, 0, 2,3)
        img=self.MIDeform2(img)
        #img=img.permute(1, 0, 2,3)
        return img



# # Initialize model and transfer it to the appropriate device
# model = MI(opt).to(device)

# # Load dataset and retrieve sample data
# dataset = DatasetFromHdf5(opt)
# ind_source, input, label, LFI = dataset[0]

# # Convert data to tensors and transfer them to the device
# ind_source = torch.from_numpy(ind_source).unsqueeze(0).to(device)
# input = input.unsqueeze(0).to(device)
# LFI = LFI.unsqueeze(0).to(device)

# # Run model forward pass
# output = model(LFI, input, opt)
# print('Output Shape:', output.shape)
