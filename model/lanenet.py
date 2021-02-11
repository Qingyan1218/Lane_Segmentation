import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import DiscriminativeLoss
from encoders import VGGEncoder
from decoders import FCNDecoder

class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()
        # 实例分割的数，4根车道和背景
        self.no_of_instances = 5
        # VGG中间重复的次数
        encode_num_blocks = 5
        # VGG输入的通道
        in_channels = [3, 64, 128, 256, 512]
        # VGG输出的通道
        out_channels = in_channels[1:] + [512]
        # 调用VGG的编码结果
        self._encoder = VGGEncoder(encode_num_blocks, in_channels, out_channels)
        # VGG中用到的输出
        decode_layers = ["pool5", "pool4", "pool3"]
        # 输出中对应的channels
        decode_channels = out_channels[:-len(decode_layers) - 1:-1]
        # 上采样用到的stride
        decode_last_stride = 8
        # 用FCN解码
        self._decoder = FCNDecoder(decode_layers, decode_channels, decode_last_stride)
        # 输出实例分割用到的卷积，输出是分割数
        self._pix_layer = nn.Conv2d(in_channels=64, out_channels=self.no_of_instances,
         kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        # 编码和解码的输出
        encode_ret = self._encoder(input_tensor)
        decode_ret = self._decoder(encode_ret)
        # decode_logits是[bs, 2, h, w]
        decode_logits = decode_ret['logits']
        # softmax后取大值所在的位置即判断是否是车道，[bs, 1, h, w]
        binary_seg_ret = torch.argmax(F.softmax(decode_logits, dim=1), dim=1, keepdim=True)
        # decode_deconv [bs, 64, h, w]
        decode_deconv = decode_ret['deconv']
        # pix_embedding [bs, 5, h, w]，代表像素点属于哪个车道
        pix_embedding = self.relu(self._pix_layer(decode_deconv))

        # 返回三个张量，
        # torch.Size([bs, 5, h, w])
        # torch.Size([bs, 1, h, w])
        # torch.Size([bs, 2, h, w])
        ret = {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': decode_logits
        }

        return ret

def compute_loss(net_output, binary_label, instance_label, device_cuda):
    # device_cuda是布尔值，表示是否用cuda
    # 两种损失不同的权重
    k_binary = 0.3
    k_instance = 0.7

    # 对于二分类的结果，直接使用交叉熵损失即可
    # 正负样本不均衡，可以采用带权重的交叉熵，也可以用focalloss
    ce_loss_fn = nn.CrossEntropyLoss()
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = ce_loss_fn(binary_seg_logits, binary_label)

    # 对于实例分割结果，使用自定义的损失，
    # 使得属于同一条车道线的像素向量距离很小，属于不同车道线的像素向量距离很大。
    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 2, 1.0, 1.0, 0.001,device_cuda)
    instance_loss, _, _, _ = ds_loss_fn(pix_embedding, instance_label, [5] * len(pix_embedding))
    
    # 加权相加
    binary_loss = binary_loss * k_binary
    instance_loss = instance_loss * k_instance
    total_loss = binary_loss + instance_loss

    # 计算iou，net_output["binary_seg_pred"]代表有车道线的地方，[bs, 1, h, w]
    out = net_output["binary_seg_pred"]
    iou = 0
    batch_size = out.size()[0]
    for i in range(batch_size):
        # PR表示预测为车道线的点的个数，nonzero输出nx2的矩阵，n表示非零值的个数，2表示像素位置
        PR = out[i].squeeze(0).nonzero(as_tuple = False).size()[0]
        # GT是真实值的个数
        GT = binary_label[i].nonzero(as_tuple = False).size()[0]
        # TP是将预测和真实相乘，即同时正确的像素个数
        TP = (out[i].squeeze(0) * binary_label[i]).nonzero(as_tuple = False).size()[0]
        # 交并比中的交即TP，并是GT+PR，但是要减去重复值TP
        union = PR + GT - TP
        iou += TP / union
    iou = iou / batch_size
    return total_loss, binary_loss, instance_loss, out, iou

if __name__ == '__main__':
    torch.cuda.empty_cache()
    x = torch.rand((4,3,32,64)).cuda()
    model = LaneNet().cuda()
    # print(model)
    out = model(x)
    for i,j in out.items():
        print(out[i].shape)

    binary_label = torch.randint(0,2,(4,32,64)).cuda()
    instance = torch.randint(0,5,(4,32,64))
    instance = F.one_hot(instance)
    instance_label = instance.permute([0,3,1,2]).cuda()

    loss = compute_loss(out,binary_label,instance_label,device_cuda=True)
    print(loss)

    """torch.Size([8, 5, 64, 64])
        torch.Size([8, 1, 64, 64])
        torch.Size([8, 2, 64, 64])"""
