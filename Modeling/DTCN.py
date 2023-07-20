import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Split_Dilation(nn.Module):
    def __init__(self,
                 in_ch=32,
                 out_ch=32,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 spl=2
                 ):
        super(Split_Dilation, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spl = spl
        self.custom_conv1 = nn.Conv2d(self.in_ch//self.spl, self.out_ch//self.spl, kernel_size, stride, padding, dilation)
        self.custom_conv2 = nn.Conv2d(self.in_ch // self.spl, self.out_ch // self.spl, kernel_size, stride, padding, dilation)

    def forward(self, input):
        input_split = torch.split(input, [self.in_ch // self.spl, self.out_ch // self.spl], dim=1)
        # Run inference
        s0 = self.custom_conv1(input_split[0])
        s1 = self.custom_conv2(input_split[1])
        # concat tensor
        out = torch.cat((s0, s1), dim=1)
        return out


# Depthwise Separable Convolution
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out



class CDALayer(nn.Module):
    def __init__(self, channel, reduction): # channel = 32, reduction = 16
        super(CDALayer, self).__init__()
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Feature Channel Rescale
        self.conv0_1 = nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv0_2 = nn.Conv2d(channel // reduction, channel // reduction, 1, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv0_3 = nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False)

        # 1 X 1 Convolution inside Skip Connection
        self.conv01 = nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False)
        self.conv02 = nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False)
        self.conv03 = nn.Conv2d(channel,  channel, 1, padding=0, bias=False)
        self.conv12 = nn.Conv2d(channel // reduction, channel // reduction, 1, padding=0, bias=False)
        self.conv13 = nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False)
        self.conv23 = nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False)
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y01 = self.conv01(y)
        y02 = self.conv02(y)
        y03 = self.conv03(y)
        y = self.conv0_1(y)
        y = self.relu0(y+y01)
        y12 = self.conv12(y)
        y13 = self.conv13(y)
        y = self.conv0_2(y)
        y = self.relu1(y + y02 + y12)
        y23 = self.conv23(y)
        y = self.conv0_3(y)
        y = self.sigmoid(y + y03 + y13 + y23)
        return x * y

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, circular_padding=False, cat=True):
        super(UNet, self).__init__()
        self.name = 'unet'
        self.cat = cat

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=32, circular_padding=circular_padding)
        self.Conv2 = conv_block(ch_in=32, ch_out=32)
        self.Conv3 = conv_block(ch_in=32, ch_out=64)
        self.md = MD(64)
        self.Up3 = up_conv(ch_in=64, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=96, ch_out=32)

        self.Up2 = up_conv(ch_in=32, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)  # 16
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # 32
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # 64
        x3 = self.md(x3)
        d3 = self.Up3(x3)
        d3 = F.interpolate(d3, x2.size()[2:], mode='bilinear', align_corners=False)
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)  # 128->64
        d2 = F.interpolate(d2, x1.size()[2:], mode='bilinear', align_corners=False)
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        out = d1

        return out, x1, x2, d3, d2

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, circular_padding=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True,
                      padding_mode='circular' if circular_padding else 'zeros'),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class MD(nn.Module):
    def __init__(self, ch):
        super(MD, self).__init__()

        self.m1 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, padding=1, dilation=1),
            nn.ReLU(),
            )
        self.m2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, padding=2, dilation=2),
            nn.ReLU(),
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, padding=4, dilation=4),
            nn.ReLU(),
            )
        self.m4 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, padding=8, dilation=8),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 1, 1, padding=1, dilation=1),
            nn.ReLU(),
        )

    def forward(self, x):
       x1 = self.m1(x)
       x2 = self.m2(x)
       x3 = self.m3(x)
       x4 = self.m4(x)
       out = self.conv(x1+x2+x3+x4)
       return out


class DTCN(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=False):
        super(DTCN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU
        self.unet =  UNet()
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            CDALayer(channel=32, reduction=16),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, 1, 1),
            CDALayer(channel=32, reduction=16),
            nn.ReLU()
            )

        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, 1, 1),
            CDALayer(channel=32, reduction=16),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, 1,1),
            CDALayer(channel=32, reduction=16),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            CDALayer(channel=32, reduction=16),
            nn.ReLU())
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1,1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )
    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
        x_list = []
        out,k1,k2,k3,k4 =self.unet(x)
        x = x - out
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            # x = self.atten1(x)
            x = F.relu(self.res_conv2(x) + resx)
            # x = self.atten2(x)
            x = F.relu(self.res_conv3(x) + resx)
            # x = self.atten3(x)
            x = F.relu(self.res_conv4(x) + resx)
            # x = self.atten4(x)
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x, out, x_list



