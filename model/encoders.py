import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models

# 删除其余两种网络，这里只使用VGG
class VGGEncoder(nn.Module):

    def __init__(self, num_blocks=5, in_channels=[3, 64, 128, 256, 512], 
        out_channels=[64, 128, 256, 512, 512]):
        super(VGGEncoder, self).__init__()
        # 定义了预训练的vgg网络
        self.pretrained_modules = models.vgg16(pretrained=True).features

        # num_blocks表示中间需要重复的次数
        self.num_blocks = num_blocks
        # in_channels和out_channels都是列表，和num_blocks对应，
        # in_channels = [3, 64, 128, 256, 512]
        # out_channels = in_channels[1:] + [512]，即[64, 128, 256, 512, 512]
        self._in_channels = in_channels
        self._out_channels = out_channels
        # conv层重复册数
        self._conv_reps = [2, 2, 3, 3, 3]
        self.net = nn.Sequential()
        self.pretrained_net = nn.Sequential()


        for i in range(num_blocks):
            # add_module中第一个参数是网络层名字，第二个参数是网络层
            self.net.add_module("block" + str(i + 1), self._encode_block(i + 1))
            self.pretrained_net.add_module("block" + str(i + 1), self._encode_pretrained_block(i + 1))

    def _encode_block(self, block_id, kernel_size=3, stride=1):
        # 传入的参数block_id加过1，所以这里要减去1
        out_channels = self._out_channels[block_id - 1]
        # padding=1，图像尺寸不变
        padding = (kernel_size - 1) // 2
        seq = nn.Sequential()

        for i in range(self._conv_reps[block_id - 1]):
            if i == 0:
                # 第一次使用输入的channel，后面都用输出的channel
                in_channels = self._in_channels[block_id - 1]
            else:
                in_channels = out_channels
            # 每一次循环都加入CBR层
            seq.add_module("conv_{}_{}".format(block_id, i + 1),
                           nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
            seq.add_module("bn_{}_{}".format(block_id, i + 1), nn.BatchNorm2d(out_channels))
            seq.add_module("relu_{}_{}".format(block_id, i + 1), nn.ReLU())
        # 一个网络块结束后加入一个pool层
        seq.add_module("maxpool" + str(block_id), nn.MaxPool2d(kernel_size=2, stride=2))
        return seq

    def _encode_pretrained_block(self, block_id):
        seq = nn.Sequential()
        # 间隔4重复一次，因为重复次数都小于4，因此仅循环一次
        for i in range(0, self._conv_reps[block_id - 1], 4):
            seq.add_module("conv_{}_{}".format(block_id, i + 1), self.pretrained_modules[i])
            seq.add_module("relu_{}_{}".format(block_id, i + 2), self.pretrained_modules[i + 1])
            seq.add_module("conv_{}_{}".format(block_id, i + 3), self.pretrained_modules[i + 2])
            seq.add_module("relu_{}_{}".format(block_id, i + 4), self.pretrained_modules[i + 3])
            seq.add_module("maxpool" + str(block_id), self.pretrained_modules[i + 4])
        return seq

    def forward(self, input_tensor):
        # 添加一个有序字典
        ret = OrderedDict()
        # 5个阶段
        X = input_tensor
        for i, block in enumerate(self.net):
            # 每一个block虽有是一个pooling层
            pool = block(X)
            # 加入字典中，让后面的lanenet根据名称提出结果
            ret["pool" + str(i + 1)] = pool
            # 下次一个block的输入即是这层的输出
            X = pool
        # model包括了三个网络，但只是用了self.net，其余的干什么用？
        return ret



if __name__ == '__main__':
    torch.cuda.empty_cache()
    x = torch.rand((8,3,64,64)).cuda()
    model = VGGEncoder().cuda()
    # print(model)
    out = model(x)
    for i,j in out.items():
        print(out[i].shape)
    """torch.Size([8, 64, 32, 32])
    torch.Size([8, 128, 16, 16])
    torch.Size([8, 256, 8, 8])
    torch.Size([8, 512, 4, 4])
    torch.Size([8, 512, 2, 2])"""
