import torch
from backbone import resnet
import numpy as np

# 定义一个conv,bn,relu的类，方便调用
class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(100, 56, 4), use_aux=True):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        # tusimple: cls_dim = (100, 56, 4)
        # 网格数量，每条车道线分段数，车道线数量
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        # 默认情况下，np.prod计算所有元素的乘积
        self.total_dim = np.prod(cls_dim)

        # 输入尺寸：[n,c,w,h]
        # 输出尺寸：[w+1,56,4]
        self.model = resnet(backbone, pretrained=pretrained)

        # 如果使用辅助训练
        if self.use_aux:
            # 如果是resnet50之后的，要先从512降到128
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )

            # 如果是resnet50之后的，要先从1024降到128
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )

            # # 如果是resnet50之后的，要先从2048降到128
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )

            # 之前的各层节后和，128*3=384，因此输入通道是384
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1,1)

            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18'] else torch.nn.Conv2d(2048,8,1)
        initialize_weights(self.cls)

    def forward(self, x):
        # 以resnet50为例
        # x2:torch.Size([n, 512, 36, 100]),1/8
        # x3:torch.Size([n, 1024, 18, 50]),1/16
        # fea:torch.Size([n, 2048, 9, 25]),1/32
        x2,x3,fea = self.model(x)

        if self.use_aux:
            # 将resnet得到的3个特征图分别进行上采样到1/8后cancat
            # 最终的输出通道数等于车道线的数量+1，即背景
            # x2: torch.Size([n, 128, 36, 100])
            x2 = self.aux_header2(x2)
            # x3: torch.Size([n, 128, 18, 50])
            x3 = self.aux_header3(x3)
            # x3_interpolate: torch.Size([n, 128, 36, 100])
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear',align_corners=False)
            # x4: torch.Size([n, 128, 9, 25])
            x4 = self.aux_header4(fea)
            # x3_interpolate: torch.Size([n, 128, 36, 100])
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear',align_corners=False)
            # aug_seg: torch.Size([n, 384, 36, 100])
            aux_seg = torch.cat([x2,x3,x4],dim=1)
            # aug_seg: torch.Size([n, 5, 36, 100])
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        # self.pool(fea).shape: torch.Size([n, 8, 9, 25])
        # fea: torch.Size([n, 1800])
        fea = self.pool(fea).view(-1, 1800)

        # group_cls：将特征图通过全连接层后再变回和label一样的尺寸
        # group_cls: torch.Size([n, 100, 56, 4])
        group_cls = self.cls(fea).view(-1, *self.cls_dim)

        if self.use_aux:
            # torch.Size([n, 100, 56, 4])
            # torch.Size([n, 5, 36, 100])
            return group_cls, aux_seg
        # torch.Size([n, 100, 56, 4])
        return group_cls


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)


if __name__=='__main__':
    from torchsummary import summary
    x = torch.rand((4,3,288,800)).cuda()
    model =parsingNet(backbone='18').cuda()
    summary(model,(3,288,800))
    y1,y2 = model(x)
    print(f'y1.shape: {y1.shape},y2.shape: {y2.shape}')