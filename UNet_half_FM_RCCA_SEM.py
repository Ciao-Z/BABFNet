import torch
import torch.nn as nn
import torch.nn.functional as F

mid_channel = 32


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        #self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        
        self.Dconv1 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, dilation=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))

        self.Dconv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=2, dilation=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))        

        self.Dconv3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=5, dilation=5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))        

        self.Dconv4 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=7, dilation=7),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))       

    def forward(self, x):
        #avg_out = torch.mean(x, dim=1, keepdim=True)
        #max_out, _ = torch.max(x, dim=1, keepdim=True)
        #x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x1 = self.Dconv1(x)
        x2 = self.Dconv2(x)
        x3 = self.Dconv3(x)
        x4 = self.Dconv4(x)
        x_ = self.conv2(torch.cat([x1,x2,x3,x4], dim=1))
        return x_


def INF(B,H,W):
    '''
    生成(B*W,H,H)大小的对角线为inf的三维矩阵
    Parameters
    ----------
    B: batch
    H: height
    W: width
    '''
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CC_module(nn.Module):
    
    def __init__(self, in_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channels of input
        '''
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim//4, out_channels=in_dim//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
          
    def forward(self, x, y):

        m_batchsize, _, height, width = x.size()
        
        proj_query = self.query_conv(x) #size = (b,c2,h,w), c1 = in_dim, c2 = c1 // 8
        
        #size = (b*w, h, c2)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1) 
        
        #size = (b*h, w, c2)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        
        proj_key = self.key_conv(y) #size = (b,c2,h,w)
        
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) #size = (b*w,c2,h) 
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) #size = (b*h,c2,w)
        
        proj_value = self.value_conv(x) #size = (b,c1,h,w)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) #size = (b*w,c1,h)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) #size = (b*h,c1,w)
        
        #size = (b*w, h,h) ,其中[:,i,j]表示Q所有W的第Hi行的所有通道值与K上所有W的第Hj列的所有通道值的向量乘积 
        energy_H = torch.bmm(proj_query_H, proj_key_H)
        
        #size = (b,h,w,h) #这里为什么加 INF并没有理解
        energy_H = (energy_H + self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        
        #size = (b*h,w,w),其中[:,i,j]表示Q所有H的第Wi行的所有通道值与K上所有H的第Wj列的所有通道值的向量乘积
        energy_W = torch.bmm(proj_query_W, proj_key_W)
        energy_W = energy_W.view(m_batchsize,height,width,width) #size = (b,h,w,w)
        
        concate = self.softmax(torch.cat([energy_H, energy_W], 3)) #size = (b,h,w,h+w) #softmax归一化
        #concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height) #size = (b*w,h,h)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width) #size = (b*h,w,w)
        
        #size = (b*w,c1,h) #[:,i,j]表示V所有W的第Ci行通道上的所有H 与att_H的所有W的第Hj列的h权重的乘积  
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1))
        out_H = out_H.view(m_batchsize,width,-1,height).permute(0,2,3,1)  #size = (b,c1,h,w)
        
        #size = (b*h,c1,w) #[:,i,j]表示V所有H的第Ci行通道上的所有W 与att_W的所有H的第Wj列的W权重的乘积  
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1))
        out_W = out_W.view(m_batchsize,height,-1,width).permute(0,2,1,3) #size = (b,c1,h,w)
        #print(out_H.size(),out_W.size())
        
        return self.gamma*(out_H + out_W) + x



class GatedSpatialConv2d(nn.Module):
    def __init__(self, in_channels):

        super(GatedSpatialConv2d, self).__init__()

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(2 * in_channels),
            nn.Conv2d(2 * in_channels, 2 * in_channels, 1),
            nn.ReLU(), 
            nn.Conv2d(2 * in_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1)) 
        return input_features


class ResBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x = self.relu(x + x1)
        return x


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


class fuseGFFConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fuseGFFConvBlock, self).__init__()
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
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        #self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #avg_out = torch.mean(x, dim=1, keepdim=True)
        #max_out, _ = torch.max(x, dim=1, keepdim=True)
        #x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class UNet_half_FM_RCCA_SEM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_half_FM_RCCA_SEM, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.gff1_conv1 = nn.Conv2d(32, mid_channel, 1)
        self.gff1_conv2 = SpatialAttention()
        self.gff1_fuse = fuseGFFConvBlock(mid_channel, mid_channel)
        self.gff1_pool = nn.MaxPool2d(1)
        self.gff1_out = nn.Conv2d(mid_channel, 32, 1)
		
        self.down1 = down(32, 64)
        self.gff2_conv1 = nn.Conv2d(64, mid_channel, 1)
        self.gff2_conv2 = SpatialAttention()
        self.gff2_fuse = fuseGFFConvBlock(mid_channel, mid_channel)
        self.gff2_pool = nn.MaxPool2d(2)
        self.gff2_out = nn.Conv2d(mid_channel, 64, 1)
		
		
        self.down2 = down(64, 128)
        self.gff3_conv1 = nn.Conv2d(128, mid_channel, 1)
        self.gff3_conv2 = SpatialAttention()
        self.gff3_fuse = fuseGFFConvBlock(mid_channel, mid_channel)
        self.gff3_pool = nn.MaxPool2d(4)
        self.gff3_out = nn.Conv2d(mid_channel, 128, 1)
		
		
        self.down3 = down(128, 256)
        self.gff4_conv1 = nn.Conv2d(256, mid_channel, 1)
        self.gff4_conv2 = SpatialAttention()
        self.gff4_fuse = fuseGFFConvBlock(mid_channel, mid_channel)
        self.gff4_pool = nn.MaxPool2d(8)
        self.gff4_out = nn.Conv2d(mid_channel, 256, 1)


        self.down4 = down(256, 256)
		
		
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        
        self.dsn2 = nn.Conv2d(64, 16, 1)
        self.dsn3 = nn.Conv2d(128, 16, 1)
        self.dsn4 = nn.Conv2d(256, 8, 1)
        
        self.res1 = ResBlock(32, 32)
        self.res1_conv = nn.Conv2d(32, 16, 1)
        self.gate1 = GatedSpatialConv2d(16)
        
        self.res2 = ResBlock(16, 16)
        self.res2_conv = nn.Conv2d(16, 16, 1)
        self.gate2 = GatedSpatialConv2d(16)
        
        self.res3 = ResBlock(16, 16)
        self.res3_conv = nn.Conv2d(16, 8, 1)
        self.gate3 = GatedSpatialConv2d(8)
        
        self.CCA1 = CC_module(32)
        self.CCA2 = CC_module(32)
        
        self.fuse = nn.Conv2d(8, 1, 1)

        self.maxpool_1 = nn.MaxPool2d(2)
        self.maxpool_2 = nn.MaxPool2d(2)
        
        
        self.sa = SpatialAttention()      
        self.cw = nn.Conv2d(2, 1, 1)    

        self.sigmoid = nn.Sigmoid()   		

		
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x_size = x.size()
        
        # GFF_att_our
        x1 = self.inc(x)
        x2 = self.down1(x1)		
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)		      
        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)

        # FM
        s2 = F.interpolate(self.dsn2(x2), x_size[2:], mode="bilinear", align_corners=True)
        s3 = F.interpolate(self.dsn3(x3), x_size[2:], mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(x4), x_size[2:], mode='bilinear', align_corners=True)
        #s_up4 = F.interpolate(self.dsn_up4(x_up4), x_size[2:], mode='bilinear', align_corners=True)
        
        m1f = F.interpolate(x1, x_size[2:], mode='bilinear', align_corners=True)
        
        cs = self.res1(m1f)
        cs = self.res1_conv(cs)
        cs = self.gate1(cs, s2)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        
        cs = self.res2(cs)
        cs = self.res2_conv(cs)
        cs = self.gate2(cs, s3)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        
        cs = self.res3(cs)
        cs = self.res3_conv(cs)
        cs = self.gate3(cs, s4)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)

        cs1 = self.fuse(cs)
        cs1 = F.interpolate(cs1, x_size[2:], mode='bilinear', align_corners=True)
        att = self.sa(x_up4)
        edge_out = self.cw(torch.cat([cs1,att], dim=1))
        
        edge_out = self.sigmoid(edge_out)	
        
        #RCCA
        x_up4 = self.maxpool_1(x_up4)
        cs = self.maxpool_2(cs)
        out_feature = self.CCA1(x_up4, cs)
        out_feature1 = self.CCA2(out_feature, cs)	
        out_feature1 = F.interpolate(out_feature1, x_size[2:], mode='bilinear', align_corners=True)
        
        x = self.outc(out_feature1)
        return torch.sigmoid(x), edge_out.squeeze(1)

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p






