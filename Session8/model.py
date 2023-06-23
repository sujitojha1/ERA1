import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1

def normalizationFx(method, out_channels):

    if method =='BN':
        return nn.BatchNorm2d(out_channels)
    else:
        group = 4 if method == 'GN' else 1
        return nn.GroupNorm(group, out_channels)
    
class Net(nn.Module):
    def __init__(self, normalizationMethod='BN'):
        super(Net, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,16),
        ) 

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,16),
            #nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1
        self.transitionblock = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,128),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,32),
            #nn.Dropout(dropout_value)
        ) 

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False),
            normalizationFx(normalizationMethod,32),
        )

        # CONVOLUTION BLOCK 3       
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.ReLU(),
            normalizationFx(normalizationMethod,128),
            #nn.Dropout(dropout_value)
        ) #o/p size = 64*8*8 RF = 26

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0, bias=False),
            normalizationFx(normalizationMethod,128),
        )
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) 

        self.linear = nn.Linear(128, 10)
        # self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x1 = self.convblock1(x)

        x2 = self.convblock2(x1)
        x3 = x2 + x1

        x4 = self.transitionblock(x3)

        x5 = self.convblock4(x4)
        x6 = x5 + self.shortcut1(x4)

        x7 = self.transitionblock(x6)

        x8 = self.convblock7(x7)
        x9 = x8 + self.shortcut2(x7)

        out = self.gap(x9)        
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out