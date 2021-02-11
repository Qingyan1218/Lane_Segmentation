import torch
import torchvision
import torch.nn.modules

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        # VGG输入是n,3,224,224
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        # model[33]：MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # 取消一个maxpool，特征图大小维持在[n,512,28,28]
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        # 去掉一个maxpool层，输出是[n,512,28,28]
        return self.model(x)

class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        # 7x7，stride=2的卷积，[n,64,112,112]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # maxpool,特征图[n,64,56,56]
        x = self.maxpool(x)
        # 经过layer1，特征图变为[n, 256, 56, 56]
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # x2.shape: torch.Size([n, 512, 28, 28]),
        # x3.shape: torch.Size([n, 1024, 14, 14]),
        # x4.shape: torch.Size([n, 2048, 7, 7])
        return x2,x3,x4

if __name__=='__main__':
    from torchsummary import summary
    x = torch.rand((8,3,224,224)).cuda()
    # model = vgg16bn().cuda()
    # summary(model.cuda(), (3, 224, 224))
    # y = model(x)
    # print(y.shape)
    model = resnet('50').cuda()
    summary(model.cuda(),(3,224,224))
    y1,y2,y3 = model(x)
    print(f'y1.shape: {y1.shape},\ny2.shape: {y2.shape},\ny3.shape: {y3.shape}')
