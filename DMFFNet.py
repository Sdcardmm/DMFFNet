import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.up(x)

class PRMModule(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.conv_maxpool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_chan, in_chan, 1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True)
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, 1, bias=False),
            nn.Conv2d(in_chan, in_chan, 3, padding=1, groups=in_chan),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True)
        )
        self.fuConv = nn.Sequential(
            nn.Conv2d(in_chan * 2, in_chan, 3, padding=1, bias=True),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.conv_maxpool(x)
        out2 = self.dwconv(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.fuConv(out)
        return out + x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    def __init__(self, in_chan, out_chan, rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DMFFNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        n1 = 16
        self.filters = [n1, n1*2, n1*4, n1*8, n1*16]

        self.Conv1 = conv_block(in_ch, self.filters[0])
        self.maxpool = nn.MaxPool2d(2, 2)
        self.Conv2 = conv_block(self.filters[0], self.filters[1])
        self.Conv3 = conv_block(self.filters[1], self.filters[2])
        self.Conv4 = conv_block(self.filters[2], self.filters[3])
        self.Conv5 = conv_block(self.filters[3], self.filters[4])

        self.maxDw1 = PRMModule(self.filters[0])
        self.maxDw2 = PRMModule(self.filters[1])
        self.maxDw3 = PRMModule(self.filters[2])
        self.maxDw4 = PRMModule(self.filters[3])
        self.maxDw5 = PRMModule(self.filters[4])

        self.D_bott1 = Bottleneck(in_ch, self.filters[0], 2)
        self.D_bott2 = Bottleneck(self.filters[0], self.filters[1], 4)
        self.D_bott3 = Bottleneck(self.filters[1], self.filters[2], 8)
        self.avgpool = nn.AvgPool2d(2, 2)

        self.se_layer = SELayer(channel=self.filters[0] + self.filters[2])
        self.classifier = nn.Sequential(
            nn.Linear(self.filters[0] + self.filters[2], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, out_ch)
        )

        self.up4 = up_conv(self.filters[4], self.filters[3])
        self.up3 = up_conv(self.filters[3], self.filters[2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, out_features=False):
        e1 = self.maxDw1(self.Conv1(x))
        e2 = self.maxDw2(self.Conv2(self.maxpool(e1)))
        e3 = self.maxDw3(self.Conv3(self.maxpool(e2)))
        e4 = self.maxDw4(self.Conv4(self.maxpool(e3)))
        e5 = self.maxDw5(self.Conv5(self.maxpool(e4)))

        d1 = self.D_bott1(x)
        d2 = self.D_bott2(self.avgpool(d1))
        d3 = self.D_bott3(self.avgpool(d2))

        d4 = self.up4(e5)
        d3 = self.up3(d4)
        d1_up = F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False)

        concat_feat = torch.cat([e1, d1_up], dim=1)
        concat_feat = self.se_layer(concat_feat)

        pooled = self.global_pool(concat_feat).flatten(1)
        out = self.classifier(pooled)

        return concat_feat if out_features else out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DMFFNet(in_ch=1, out_ch=2).to(device)
    x = torch.randn(8, 1, 256, 256).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")