import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        # Tạo các DenseLayer cho DenseBlock
        for i in range(num_layers):
            self.layers.append(DenseLayer(num_input_features + i * growth_rate, growth_rate))
    
    def forward(self, x):
        # Truyền tín hiệu qua từng layer của DenseBlock
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)  # Concatenate đầu ra vào đầu vào
        return x

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate):
        super(DenseLayer, self).__init__()
        
        # 1x1 convolution (bottleneck layer)
        self.bottleneck = nn.Conv2d(num_input_features, 4 * growth_rate, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(4 * growth_rate)
        
        # 3x3 convolution
        self.conv = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(growth_rate)
    
    def forward(self, x):
        bottleneck_out = F.relu(self.batch_norm1(self.bottleneck(x)))
        new_features = F.relu(self.batch_norm2(self.conv(bottleneck_out)))
        return new_features

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_output_features)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = F.relu(self.batch_norm(self.conv(x)))
        x = self.avg_pool(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, num_blocks=[6, 12, 24, 16], num_classes=1000):
        super(DenseNet121, self).__init__()
        
        # Initial Convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Tạo các DenseBlocks và TransitionLayers
        num_input_features = 64
        self.dense_blocks = []
        for i, num_layers in enumerate(num_blocks):
            dense_block = DenseBlock(num_layers, num_input_features, growth_rate)
            self.dense_blocks.append(dense_block)
            num_input_features += num_layers * growth_rate
            if i != len(num_blocks) - 1:  # không tạo TransitionLayer cho block cuối cùng
                transition = TransitionLayer(num_input_features, num_input_features // 2)
                self.dense_blocks.append(transition)
                num_input_features = num_input_features // 2
        
        self.dense_blocks = nn.ModuleList(self.dense_blocks)
        
        # Fully Connected Layer (classifier)
        self.fc = nn.Linear(num_input_features, num_classes)
    
    def forward(self, x):
        # Tiến hành các bước theo từng phần của mô hình
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        
        # Pass qua các DenseBlocks và TransitionLayers
        for layer in self.dense_blocks:
            x = layer(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Lớp phân loại cuối
        x = self.fc(x)
        return x

# model = DenseNet121()
# model.to("cuda")

# summary(model, (3, 224, 224), 128)

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1        [128, 64, 112, 112]           9,408
#        BatchNorm2d-2        [128, 64, 112, 112]             128
#               ReLU-3        [128, 64, 112, 112]               0
#          MaxPool2d-4          [128, 64, 56, 56]               0
#             Conv2d-5         [128, 128, 56, 56]           8,192
#        BatchNorm2d-6         [128, 128, 56, 56]             256
#             Conv2d-7          [128, 32, 56, 56]          36,864
#        BatchNorm2d-8          [128, 32, 56, 56]              64
#         DenseLayer-9          [128, 32, 56, 56]               0
#            Conv2d-10         [128, 128, 56, 56]          12,288
#       BatchNorm2d-11         [128, 128, 56, 56]             256
#            Conv2d-12          [128, 32, 56, 56]          36,864
#       BatchNorm2d-13          [128, 32, 56, 56]              64
#        DenseLayer-14          [128, 32, 56, 56]               0
#            Conv2d-15         [128, 128, 56, 56]          16,384
#       BatchNorm2d-16         [128, 128, 56, 56]             256
#            Conv2d-17          [128, 32, 56, 56]          36,864
#       BatchNorm2d-18          [128, 32, 56, 56]              64
#        DenseLayer-19          [128, 32, 56, 56]               0
#            Conv2d-20         [128, 128, 56, 56]          20,480
#       BatchNorm2d-21         [128, 128, 56, 56]             256
#            Conv2d-22          [128, 32, 56, 56]          36,864
#       BatchNorm2d-23          [128, 32, 56, 56]              64
#        DenseLayer-24          [128, 32, 56, 56]               0
#            Conv2d-25         [128, 128, 56, 56]          24,576
#       BatchNorm2d-26         [128, 128, 56, 56]             256
#            Conv2d-27          [128, 32, 56, 56]          36,864
#       BatchNorm2d-28          [128, 32, 56, 56]              64
#        DenseLayer-29          [128, 32, 56, 56]               0
#            Conv2d-30         [128, 128, 56, 56]          28,672
#       BatchNorm2d-31         [128, 128, 56, 56]             256
#            Conv2d-32          [128, 32, 56, 56]          36,864
#       BatchNorm2d-33          [128, 32, 56, 56]              64
#        DenseLayer-34          [128, 32, 56, 56]               0
#        DenseBlock-35         [128, 256, 56, 56]               0
#            Conv2d-36         [128, 128, 56, 56]          32,768
#       BatchNorm2d-37         [128, 128, 56, 56]             256
#         AvgPool2d-38         [128, 128, 28, 28]               0
#   TransitionLayer-39         [128, 128, 28, 28]               0
#            Conv2d-40         [128, 128, 28, 28]          16,384
#       BatchNorm2d-41         [128, 128, 28, 28]             256
#            Conv2d-42          [128, 32, 28, 28]          36,864
#       BatchNorm2d-43          [128, 32, 28, 28]              64
#        DenseLayer-44          [128, 32, 28, 28]               0
#            Conv2d-45         [128, 128, 28, 28]          20,480
#       BatchNorm2d-46         [128, 128, 28, 28]             256
#            Conv2d-47          [128, 32, 28, 28]          36,864
#       BatchNorm2d-48          [128, 32, 28, 28]              64
#        DenseLayer-49          [128, 32, 28, 28]               0
#            Conv2d-50         [128, 128, 28, 28]          24,576
#       BatchNorm2d-51         [128, 128, 28, 28]             256
#            Conv2d-52          [128, 32, 28, 28]          36,864
#       BatchNorm2d-53          [128, 32, 28, 28]              64
#        DenseLayer-54          [128, 32, 28, 28]               0
#            Conv2d-55         [128, 128, 28, 28]          28,672
#       BatchNorm2d-56         [128, 128, 28, 28]             256
#            Conv2d-57          [128, 32, 28, 28]          36,864
#       BatchNorm2d-58          [128, 32, 28, 28]              64
#        DenseLayer-59          [128, 32, 28, 28]               0
#            Conv2d-60         [128, 128, 28, 28]          32,768
#       BatchNorm2d-61         [128, 128, 28, 28]             256
#            Conv2d-62          [128, 32, 28, 28]          36,864
#       BatchNorm2d-63          [128, 32, 28, 28]              64
#        DenseLayer-64          [128, 32, 28, 28]               0
#            Conv2d-65         [128, 128, 28, 28]          36,864
#       BatchNorm2d-66         [128, 128, 28, 28]             256
#            Conv2d-67          [128, 32, 28, 28]          36,864
#       BatchNorm2d-68          [128, 32, 28, 28]              64
#        DenseLayer-69          [128, 32, 28, 28]               0
#            Conv2d-70         [128, 128, 28, 28]          40,960
#       BatchNorm2d-71         [128, 128, 28, 28]             256
#            Conv2d-72          [128, 32, 28, 28]          36,864
#       BatchNorm2d-73          [128, 32, 28, 28]              64
#        DenseLayer-74          [128, 32, 28, 28]               0
#            Conv2d-75         [128, 128, 28, 28]          45,056
#       BatchNorm2d-76         [128, 128, 28, 28]             256
#            Conv2d-77          [128, 32, 28, 28]          36,864
#       BatchNorm2d-78          [128, 32, 28, 28]              64
#        DenseLayer-79          [128, 32, 28, 28]               0
#            Conv2d-80         [128, 128, 28, 28]          49,152
#       BatchNorm2d-81         [128, 128, 28, 28]             256
#            Conv2d-82          [128, 32, 28, 28]          36,864
#       BatchNorm2d-83          [128, 32, 28, 28]              64
#        DenseLayer-84          [128, 32, 28, 28]               0
#            Conv2d-85         [128, 128, 28, 28]          53,248
#       BatchNorm2d-86         [128, 128, 28, 28]             256
#            Conv2d-87          [128, 32, 28, 28]          36,864
#       BatchNorm2d-88          [128, 32, 28, 28]              64
#        DenseLayer-89          [128, 32, 28, 28]               0
#            Conv2d-90         [128, 128, 28, 28]          57,344
#       BatchNorm2d-91         [128, 128, 28, 28]             256
#            Conv2d-92          [128, 32, 28, 28]          36,864
#       BatchNorm2d-93          [128, 32, 28, 28]              64
#        DenseLayer-94          [128, 32, 28, 28]               0
#            Conv2d-95         [128, 128, 28, 28]          61,440
#       BatchNorm2d-96         [128, 128, 28, 28]             256
#            Conv2d-97          [128, 32, 28, 28]          36,864
#       BatchNorm2d-98          [128, 32, 28, 28]              64
#        DenseLayer-99          [128, 32, 28, 28]               0
#       DenseBlock-100         [128, 512, 28, 28]               0
#           Conv2d-101         [128, 256, 28, 28]         131,072
#      BatchNorm2d-102         [128, 256, 28, 28]             512
#        AvgPool2d-103         [128, 256, 14, 14]               0
#  TransitionLayer-104         [128, 256, 14, 14]               0
#           Conv2d-105         [128, 128, 14, 14]          32,768
#      BatchNorm2d-106         [128, 128, 14, 14]             256
#           Conv2d-107          [128, 32, 14, 14]          36,864
#      BatchNorm2d-108          [128, 32, 14, 14]              64
#       DenseLayer-109          [128, 32, 14, 14]               0
#           Conv2d-110         [128, 128, 14, 14]          36,864
#      BatchNorm2d-111         [128, 128, 14, 14]             256
#           Conv2d-112          [128, 32, 14, 14]          36,864
#      BatchNorm2d-113          [128, 32, 14, 14]              64
#       DenseLayer-114          [128, 32, 14, 14]               0
#           Conv2d-115         [128, 128, 14, 14]          40,960
#      BatchNorm2d-116         [128, 128, 14, 14]             256
#           Conv2d-117          [128, 32, 14, 14]          36,864
#      BatchNorm2d-118          [128, 32, 14, 14]              64
#       DenseLayer-119          [128, 32, 14, 14]               0
#           Conv2d-120         [128, 128, 14, 14]          45,056
#      BatchNorm2d-121         [128, 128, 14, 14]             256
#           Conv2d-122          [128, 32, 14, 14]          36,864
#      BatchNorm2d-123          [128, 32, 14, 14]              64
#       DenseLayer-124          [128, 32, 14, 14]               0
#           Conv2d-125         [128, 128, 14, 14]          49,152
#      BatchNorm2d-126         [128, 128, 14, 14]             256
#           Conv2d-127          [128, 32, 14, 14]          36,864
#      BatchNorm2d-128          [128, 32, 14, 14]              64
#       DenseLayer-129          [128, 32, 14, 14]               0
#           Conv2d-130         [128, 128, 14, 14]          53,248
#      BatchNorm2d-131         [128, 128, 14, 14]             256
#           Conv2d-132          [128, 32, 14, 14]          36,864
#      BatchNorm2d-133          [128, 32, 14, 14]              64
#       DenseLayer-134          [128, 32, 14, 14]               0
#           Conv2d-135         [128, 128, 14, 14]          57,344
#      BatchNorm2d-136         [128, 128, 14, 14]             256
#           Conv2d-137          [128, 32, 14, 14]          36,864
#      BatchNorm2d-138          [128, 32, 14, 14]              64
#       DenseLayer-139          [128, 32, 14, 14]               0
#           Conv2d-140         [128, 128, 14, 14]          61,440
#      BatchNorm2d-141         [128, 128, 14, 14]             256
#           Conv2d-142          [128, 32, 14, 14]          36,864
#      BatchNorm2d-143          [128, 32, 14, 14]              64
#       DenseLayer-144          [128, 32, 14, 14]               0
#           Conv2d-145         [128, 128, 14, 14]          65,536
#      BatchNorm2d-146         [128, 128, 14, 14]             256
#           Conv2d-147          [128, 32, 14, 14]          36,864
#      BatchNorm2d-148          [128, 32, 14, 14]              64
#       DenseLayer-149          [128, 32, 14, 14]               0
#           Conv2d-150         [128, 128, 14, 14]          69,632
#      BatchNorm2d-151         [128, 128, 14, 14]             256
#           Conv2d-152          [128, 32, 14, 14]          36,864
#      BatchNorm2d-153          [128, 32, 14, 14]              64
#       DenseLayer-154          [128, 32, 14, 14]               0
#           Conv2d-155         [128, 128, 14, 14]          73,728
#      BatchNorm2d-156         [128, 128, 14, 14]             256
#           Conv2d-157          [128, 32, 14, 14]          36,864
#      BatchNorm2d-158          [128, 32, 14, 14]              64
#       DenseLayer-159          [128, 32, 14, 14]               0
#           Conv2d-160         [128, 128, 14, 14]          77,824
#      BatchNorm2d-161         [128, 128, 14, 14]             256
#           Conv2d-162          [128, 32, 14, 14]          36,864
#      BatchNorm2d-163          [128, 32, 14, 14]              64
#       DenseLayer-164          [128, 32, 14, 14]               0
#           Conv2d-165         [128, 128, 14, 14]          81,920
#      BatchNorm2d-166         [128, 128, 14, 14]             256
#           Conv2d-167          [128, 32, 14, 14]          36,864
#      BatchNorm2d-168          [128, 32, 14, 14]              64
#       DenseLayer-169          [128, 32, 14, 14]               0
#           Conv2d-170         [128, 128, 14, 14]          86,016
#      BatchNorm2d-171         [128, 128, 14, 14]             256
#           Conv2d-172          [128, 32, 14, 14]          36,864
#      BatchNorm2d-173          [128, 32, 14, 14]              64
#       DenseLayer-174          [128, 32, 14, 14]               0
#           Conv2d-175         [128, 128, 14, 14]          90,112
#      BatchNorm2d-176         [128, 128, 14, 14]             256
#           Conv2d-177          [128, 32, 14, 14]          36,864
#      BatchNorm2d-178          [128, 32, 14, 14]              64
#       DenseLayer-179          [128, 32, 14, 14]               0
#           Conv2d-180         [128, 128, 14, 14]          94,208
#      BatchNorm2d-181         [128, 128, 14, 14]             256
#           Conv2d-182          [128, 32, 14, 14]          36,864
#      BatchNorm2d-183          [128, 32, 14, 14]              64
#       DenseLayer-184          [128, 32, 14, 14]               0
#           Conv2d-185         [128, 128, 14, 14]          98,304
#      BatchNorm2d-186         [128, 128, 14, 14]             256
#           Conv2d-187          [128, 32, 14, 14]          36,864
#      BatchNorm2d-188          [128, 32, 14, 14]              64
#       DenseLayer-189          [128, 32, 14, 14]               0
#           Conv2d-190         [128, 128, 14, 14]         102,400
#      BatchNorm2d-191         [128, 128, 14, 14]             256
#           Conv2d-192          [128, 32, 14, 14]          36,864
#      BatchNorm2d-193          [128, 32, 14, 14]              64
#       DenseLayer-194          [128, 32, 14, 14]               0
#           Conv2d-195         [128, 128, 14, 14]         106,496
#      BatchNorm2d-196         [128, 128, 14, 14]             256
#           Conv2d-197          [128, 32, 14, 14]          36,864
#      BatchNorm2d-198          [128, 32, 14, 14]              64
#       DenseLayer-199          [128, 32, 14, 14]               0
#           Conv2d-200         [128, 128, 14, 14]         110,592
#      BatchNorm2d-201         [128, 128, 14, 14]             256
#           Conv2d-202          [128, 32, 14, 14]          36,864
#      BatchNorm2d-203          [128, 32, 14, 14]              64
#       DenseLayer-204          [128, 32, 14, 14]               0
#           Conv2d-205         [128, 128, 14, 14]         114,688
#      BatchNorm2d-206         [128, 128, 14, 14]             256
#           Conv2d-207          [128, 32, 14, 14]          36,864
#      BatchNorm2d-208          [128, 32, 14, 14]              64
#       DenseLayer-209          [128, 32, 14, 14]               0
#           Conv2d-210         [128, 128, 14, 14]         118,784
#      BatchNorm2d-211         [128, 128, 14, 14]             256
#           Conv2d-212          [128, 32, 14, 14]          36,864
#      BatchNorm2d-213          [128, 32, 14, 14]              64
#       DenseLayer-214          [128, 32, 14, 14]               0
#           Conv2d-215         [128, 128, 14, 14]         122,880
#      BatchNorm2d-216         [128, 128, 14, 14]             256
#           Conv2d-217          [128, 32, 14, 14]          36,864
#      BatchNorm2d-218          [128, 32, 14, 14]              64
#       DenseLayer-219          [128, 32, 14, 14]               0
#           Conv2d-220         [128, 128, 14, 14]         126,976
#      BatchNorm2d-221         [128, 128, 14, 14]             256
#           Conv2d-222          [128, 32, 14, 14]          36,864
#      BatchNorm2d-223          [128, 32, 14, 14]              64
#       DenseLayer-224          [128, 32, 14, 14]               0
#       DenseBlock-225        [128, 1024, 14, 14]               0
#           Conv2d-226         [128, 512, 14, 14]         524,288
#      BatchNorm2d-227         [128, 512, 14, 14]           1,024
#        AvgPool2d-228           [128, 512, 7, 7]               0
#  TransitionLayer-229           [128, 512, 7, 7]               0
#           Conv2d-230           [128, 128, 7, 7]          65,536
#      BatchNorm2d-231           [128, 128, 7, 7]             256
#           Conv2d-232            [128, 32, 7, 7]          36,864
#      BatchNorm2d-233            [128, 32, 7, 7]              64
#       DenseLayer-234            [128, 32, 7, 7]               0
#           Conv2d-235           [128, 128, 7, 7]          69,632
#      BatchNorm2d-236           [128, 128, 7, 7]             256
#           Conv2d-237            [128, 32, 7, 7]          36,864
#      BatchNorm2d-238            [128, 32, 7, 7]              64
#       DenseLayer-239            [128, 32, 7, 7]               0
#           Conv2d-240           [128, 128, 7, 7]          73,728
#      BatchNorm2d-241           [128, 128, 7, 7]             256
#           Conv2d-242            [128, 32, 7, 7]          36,864
#      BatchNorm2d-243            [128, 32, 7, 7]              64
#       DenseLayer-244            [128, 32, 7, 7]               0
#           Conv2d-245           [128, 128, 7, 7]          77,824
#      BatchNorm2d-246           [128, 128, 7, 7]             256
#           Conv2d-247            [128, 32, 7, 7]          36,864
#      BatchNorm2d-248            [128, 32, 7, 7]              64
#       DenseLayer-249            [128, 32, 7, 7]               0
#           Conv2d-250           [128, 128, 7, 7]          81,920
#      BatchNorm2d-251           [128, 128, 7, 7]             256
#           Conv2d-252            [128, 32, 7, 7]          36,864
#      BatchNorm2d-253            [128, 32, 7, 7]              64
#       DenseLayer-254            [128, 32, 7, 7]               0
#           Conv2d-255           [128, 128, 7, 7]          86,016
#      BatchNorm2d-256           [128, 128, 7, 7]             256
#           Conv2d-257            [128, 32, 7, 7]          36,864
#      BatchNorm2d-258            [128, 32, 7, 7]              64
#       DenseLayer-259            [128, 32, 7, 7]               0
#           Conv2d-260           [128, 128, 7, 7]          90,112
#      BatchNorm2d-261           [128, 128, 7, 7]             256
#           Conv2d-262            [128, 32, 7, 7]          36,864
#      BatchNorm2d-263            [128, 32, 7, 7]              64
#       DenseLayer-264            [128, 32, 7, 7]               0
#           Conv2d-265           [128, 128, 7, 7]          94,208
#      BatchNorm2d-266           [128, 128, 7, 7]             256
#           Conv2d-267            [128, 32, 7, 7]          36,864
#      BatchNorm2d-268            [128, 32, 7, 7]              64
#       DenseLayer-269            [128, 32, 7, 7]               0
#           Conv2d-270           [128, 128, 7, 7]          98,304
#      BatchNorm2d-271           [128, 128, 7, 7]             256
#           Conv2d-272            [128, 32, 7, 7]          36,864
#      BatchNorm2d-273            [128, 32, 7, 7]              64
#       DenseLayer-274            [128, 32, 7, 7]               0
#           Conv2d-275           [128, 128, 7, 7]         102,400
#      BatchNorm2d-276           [128, 128, 7, 7]             256
#           Conv2d-277            [128, 32, 7, 7]          36,864
#      BatchNorm2d-278            [128, 32, 7, 7]              64
#       DenseLayer-279            [128, 32, 7, 7]               0
#           Conv2d-280           [128, 128, 7, 7]         106,496
#      BatchNorm2d-281           [128, 128, 7, 7]             256
#           Conv2d-282            [128, 32, 7, 7]          36,864
#      BatchNorm2d-283            [128, 32, 7, 7]              64
#       DenseLayer-284            [128, 32, 7, 7]               0
#           Conv2d-285           [128, 128, 7, 7]         110,592
#      BatchNorm2d-286           [128, 128, 7, 7]             256
#           Conv2d-287            [128, 32, 7, 7]          36,864
#      BatchNorm2d-288            [128, 32, 7, 7]              64
#       DenseLayer-289            [128, 32, 7, 7]               0
#           Conv2d-290           [128, 128, 7, 7]         114,688
#      BatchNorm2d-291           [128, 128, 7, 7]             256
#           Conv2d-292            [128, 32, 7, 7]          36,864
#      BatchNorm2d-293            [128, 32, 7, 7]              64
#       DenseLayer-294            [128, 32, 7, 7]               0
#           Conv2d-295           [128, 128, 7, 7]         118,784
#      BatchNorm2d-296           [128, 128, 7, 7]             256
#           Conv2d-297            [128, 32, 7, 7]          36,864
#      BatchNorm2d-298            [128, 32, 7, 7]              64
#       DenseLayer-299            [128, 32, 7, 7]               0
#           Conv2d-300           [128, 128, 7, 7]         122,880
#      BatchNorm2d-301           [128, 128, 7, 7]             256
#           Conv2d-302            [128, 32, 7, 7]          36,864
#      BatchNorm2d-303            [128, 32, 7, 7]              64
#       DenseLayer-304            [128, 32, 7, 7]               0
#           Conv2d-305           [128, 128, 7, 7]         126,976
#      BatchNorm2d-306           [128, 128, 7, 7]             256
#           Conv2d-307            [128, 32, 7, 7]          36,864
#      BatchNorm2d-308            [128, 32, 7, 7]              64
#       DenseLayer-309            [128, 32, 7, 7]               0
#       DenseBlock-310          [128, 1024, 7, 7]               0
#           Linear-311                [128, 1000]       1,025,000
# ================================================================
# Total params: 7,915,688
# Trainable params: 7,915,688
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 73.50
# Forward/backward pass size (MB): 15494.52
# Params size (MB): 30.20
# Estimated Total Size (MB): 15598.22
# ----------------------------------------------------------------