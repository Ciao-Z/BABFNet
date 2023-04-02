import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_UNet import UNet
from .base_NestedUNet import NestedUNet
from .base_UNet3Plus import UNet3Plus
from .base_DFEM import DFEM



class MSFE(nn.Module):
    def __init__(self, in_ch):
        super(MSFE, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//4, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        
        self.Dconv1 = nn.Sequential(
            nn.Conv2d(in_ch//4, in_ch//4, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_ch//4),
            nn.ReLU(inplace=True))

        self.Dconv2 = nn.Sequential(
            nn.Conv2d(in_ch//4, in_ch//4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_ch//4),
            nn.ReLU(inplace=True))        

        self.Dconv3 = nn.Sequential(
            nn.Conv2d(in_ch//4, in_ch//4, 3, padding=5, dilation=5),
            nn.BatchNorm2d(in_ch//4),
            nn.ReLU(inplace=True))        

        self.Dconv4 = nn.Sequential(
            nn.Conv2d(in_ch//4, in_ch//4, 3, padding=7, dilation=7),
            nn.BatchNorm2d(in_ch//4),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))       

    def forward(self, x):

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


class CC_fusion(nn.Module):
    
    def __init__(self, in_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channels of input
        '''
        super(CC_fusion, self).__init__()
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


class SEM_fuse(nn.Module):
    def __init__(self, in_channels):

        super(SEM_fuse, self).__init__()

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


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class BABFNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_n=0, r=1):
        super(BABFNet, self).__init__()
        if base_n==0:
            self.base = UNet(n_channels, n_classes, r=1)
        if base_n==1:
            self.base = NestedUNet(n_channels, n_classes, r=1)
        if base_n==2:
            self.base = UNet3Plus(n_channels, n_classes, r=1)
        if base_n==3:
            self.base = DFEM(n_channels, n_classes, r=1)
        
        if r==1:
            filters = [32, 64, 128, 256, 256]
        if r==2:
            filters = [64, 128, 256, 512, 512]
        
        self.dsn2 = nn.Conv2d(filters[1], 16, 1)
        self.dsn3 = nn.Conv2d(filters[2], 16, 1)
        self.dsn4 = nn.Conv2d(filters[3], 8, 1)
        
        self.res1 = ResBlock(filters[0], 32)
        self.res1_conv = nn.Conv2d(32, 16, 1)
        self.gate1 = SEM_fuse(16)
        
        self.res2 = ResBlock(16, 16)
        self.res2_conv = nn.Conv2d(16, 16, 1)
        self.gate2 = SEM_fuse(16)
        
        self.res3 = ResBlock(16, 16)
        self.res3_conv = nn.Conv2d(16, 8, 1)
        self.gate3 = SEM_fuse(8)
        
        self.FF = CC_fusion(filters[0])
        self.SF = CC_fusion(filters[0])
        
        self.fuse = nn.Conv2d(8, 1, 1)

        self.maxpool_1 = nn.MaxPool2d(2)
        self.maxpool_2 = nn.MaxPool2d(2)
        
        self.msfe = MSFE()
        self.cw = nn.Conv2d(2, 1, 1)

        self.sigmoid = nn.Sigmoid()   		
		
        self.outc = outconv(filters[0], n_classes)

    def forward(self, x):
        x_size = x.size()
        x1, x2, x3, x4, x5, x_up4 = self.base(x)

        s2 = F.interpolate(self.dsn2(x2), x_size[2:], mode="bilinear", align_corners=True)
        s3 = F.interpolate(self.dsn3(x3), x_size[2:], mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(x4), x_size[2:], mode='bilinear', align_corners=True)
        
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
        att = self.msfe(x_up4)
        edge_out = self.cw(torch.cat([cs1,att], dim=1))
        edge_out = self.sigmoid(edge_out)	

        x_up4 = self.maxpool_1(x_up4)
        cs = self.maxpool_2(cs)
        out_feature = self.FF(x_up4, cs)
        out_feature1 = self.SF(out_feature, cs)	
        out_feature1 = F.interpolate(out_feature1, x_size[2:], mode='bilinear', align_corners=True)
        
        x = self.outc(out_feature1)
        return torch.sigmoid(x), edge_out.squeeze(1)

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p






