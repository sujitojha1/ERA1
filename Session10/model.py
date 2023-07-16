import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1

class X(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(X, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1,bias = False),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(x):
        return self.conv1(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(x):
        out = self.conv(x)
        out = self.conv(out)
        out = out + x
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Prep Layer
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 1
        self.X1 = X(in_channels=64,out_channels=128)
        self.R1 = ResBlock(in_channels=128,out_channels=128)

        # Layer 2
        self.X2 = X(in_channels=128,out_channels=256)

        # Layer 3
        self.X3 = X(in_channels=256,out_channels=512)
        self.R3 = ResBlock(in_channels=512,out_channels=512)

        # Max Pool
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2)

        # FC
        self.fc = nn.Linear(2048,10)

    def forward(x):
        batch_size = x.shape[0]

        out = self.preplayer(x)

        # Layer 1
        X = self.X1(out)
        R1 = self.R1(X)

        out = X + R1

        # Layer 2
        out = self.X2(out)

        # Layer 3
        X = self.X3(out)
        R2 = self.R3(X)  

        out = X + R2

        out = self.maxpool(out)

        # FC
        out = out.view(batch_size,-1)
        out = self.fc(out)

        return F.Softmax(out, dim=1)