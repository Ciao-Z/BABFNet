import torch
import torch.nn as nn
import torch.nn.functional as F

mid_channel = 32


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class fuseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fuseBlock, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fuse(x)
        return x

		
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        return self.sigmoid(x)


class DFEM(nn.Module):
    def __init__(self, n_channels, n_classes, r=1):
        super(DFEM, self).__init__()

        if r==1:
            filters = [32, 64, 128, 256, 256]
        if r==2:
            filters = [64, 128, 256, 512, 512]   

        self.inc = inconv(n_channels, filters[0])
        self.DFEM1_conv1 = nn.Conv2d(filters[0], mid_channel, 1)
        self.DFEM1_conv2 = SpatialAttention()
        self.DFEM1_fuse = fuseBlock(mid_channel, mid_channel)
        self.DFEM1_pool = nn.MaxPool2d(1)
        self.DFEM1_out = nn.Conv2d(mid_channel, filters[0], 1)
		
        self.down1 = down(filters[0], filters[1])
        self.DFEM2_conv1 = nn.Conv2d(filters[1], mid_channel, 1)
        self.DFEM2_conv2 = SpatialAttention()
        self.DFEM2_fuse = fuseBlock(mid_channel, mid_channel)
        self.DFEM2_pool = nn.MaxPool2d(2)
        self.DFEM2_out = nn.Conv2d(mid_channel, filters[1], 1)
		
		
        self.down2 = down(filters[1], filters[2])
        self.DFEM3_conv1 = nn.Conv2d(filters[2], mid_channel, 1)
        self.DFEM3_conv2 = SpatialAttention()
        self.DFEM3_fuse = fuseBlock(mid_channel, mid_channel)
        self.DFEM3_pool = nn.MaxPool2d(4)
        self.DFEM3_out = nn.Conv2d(mid_channel, filters[2], 1)
		
		
        self.down3 = down(filters[2], filters[3])
        self.DFEM4_conv1 = nn.Conv2d(filters[3], mid_channel, 1)
        self.DFEM4_conv2 = SpatialAttention()
        self.DFEM4_fuse = fuseBlock(mid_channel, mid_channel)
        self.DFEM4_pool = nn.MaxPool2d(8)
        self.DFEM4_out = nn.Conv2d(mid_channel, filters[3], 1)


        self.down4 = down(filters[3], filters[4])
        self.up1 = up(2*filters[3], filters[2])
        self.up2 = up(2*filters[2], filters[1])
        self.up3 = up(2*filters[1], filters[0])
        self.up4 = up(2*filters[0], filters[0])
        #self.sigmoid = nn.Sigmoid()   	
        #self.outc = outconv(filters[0], n_classes)

    def forward(self, x):
        x_size = x.size()
        
        x1 = self.inc(x)
        x1n = F.interpolate(x1, x_size[2:], mode="bilinear", align_corners=True)
        x1n = self.DFEM1_conv1(x1n)
        p1 = self.DFEM1_conv2(x1n)
		
        x2 = self.down1(x1)
        x2n = F.interpolate(x2, x_size[2:], mode="bilinear", align_corners=True)
        x2n = self.DFEM2_conv1(x2n)
        p2 = self.DFEM2_conv2(x2n)
		
        x3 = self.down2(x2)
        x3n = F.interpolate(x3, x_size[2:], mode="bilinear", align_corners=True)
        x3n = self.DFEM3_conv1(x3n)
        p3 = self.DFEM3_conv2(x3n)
		
        x4 = self.down3(x3)
        x4n = F.interpolate(x4, x_size[2:], mode="bilinear", align_corners=True)
        x4n = self.DFEM4_conv1(x4n)
        p4 = self.DFEM4_conv2(x4n)
				
        x5 = self.down4(x4)
		
        x1_DFEM = (1+p1)*x1n + (1-p1)*(g2*x2n + g3*x3n + g4*x4n)
        x2_DFEM = (1+p2)*x2n + (1-p2)*(g1*x1n + g3*x3n + g4*x4n)
        x3_DFEM = (1+p3)*x3n + (1-p3)*(g2*x2n + g1*x1n + g4*x4n)
        x4_DFEM = (1+p4)*x4n + (1-p4)*(g2*x2n + g3*x3n + g1*x1n)
		
        x1_DFEM = self.DFEM1_fuse(x1_DFEM)
        x1_DFEM = self.DFEM1_pool(x1_DFEM)
        x1_DFEM = self.DFEM1_out(x1_DFEM)
        
        x2_DFEM = self.DFEM2_fuse(x2_DFEM)
        x2_DFEM = self.DFEM2_pool(x2_DFEM)
        x2_DFEM = self.DFEM2_out(x2_DFEM)
        
        x3_DFEM = self.DFEM3_fuse(x3_DFEM)
        x3_DFEM = self.DFEM3_pool(x3_DFEM)
        x3_DFEM = self.DFEM3_out(x3_DFEM)
        
        x4_DFEM = self.DFEM4_fuse(x4_DFEM)
        x4_DFEM = self.DFEM4_pool(x4_DFEM)
        x4_DFEM = self.DFEM4_out(x4_DFEM)   
        
        x_up1 = self.up1(x5, x4_DFEM)
        x_up2 = self.up2(x_up1, x3_DFEM)
        x_up3 = self.up3(x_up2, x2_DFEM)
        x_up4 = self.up4(x_up3, x1_DFEM)

        #x = self.outc(x_up4)
        return x1, x2, x3, x4, x5, x_up4

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p






