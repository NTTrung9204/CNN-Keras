import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN_Attention(nn.Module):
    def __init__(self):
        super(CNN_Attention, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        self.fc1 = nn.Linear(512 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1)
        
        batch_size, channels, height, width = x.size()
        
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)
        
        attn_output, _ = self.attn(x, x, x)
        
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, height, width)
        
        x = attn_output.reshape(batch_size, -1)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)

        x = self.fc2(x)
        
        return x

# model = CNN_Attention()
# model.to("cuda")

# summary(model, (3, 224, 224), 128)

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1        [128, 64, 112, 112]           1,792
#        BatchNorm2d-2        [128, 64, 112, 112]             128
#             Conv2d-3         [128, 128, 56, 56]          73,856
#        BatchNorm2d-4         [128, 128, 56, 56]             256
#             Conv2d-5         [128, 256, 28, 28]         295,168
#        BatchNorm2d-6         [128, 256, 28, 28]             512
#             Conv2d-7         [128, 512, 14, 14]       1,180,160
#        BatchNorm2d-8         [128, 512, 14, 14]           1,024
# MultiheadAttention-9  [[-1, 2, 512], [-1, 2, 2]]               0
#            Linear-10                 [128, 128]      12,845,184
#            Linear-11                   [128, 2]             258
# ================================================================
# Total params: 14,398,338
# Trainable params: 14,398,338
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 73.50
# Forward/backward pass size (MB): 2940.16
# Params size (MB): 54.93
# Estimated Total Size (MB): 3068.58
# ----------------------------------------------------------------