import torch
import torch.nn as nn

# 这里只是用FCN网络
class FCNDecoder(nn.Module):
    def __init__(self, decode_layers, decode_channels=[], decode_last_stride=8):
        super(FCNDecoder, self).__init__()
        # encoder的网络输出通道是512
        # decode_layers = ["pool5", "pool4", "pool3"]
        # decode_channels = out_channels[:-len(decode_layers) - 1:-1]
        # 即out_channels[:-3 - 1:-1]，[64, 128, 256, 512, 512][:-4:-1]
        # 即[512,512,256]
        # decode_last_stride = 8
        self._decode_channels = [512, 256]
        self._out_channel = 64
        self._decode_layers = decode_layers

        modules = []
        # 此处要用到nn.Modulist，否则无法加入cuda
        for _ch in self._decode_channels:
            modules.append(nn.Conv2d(_ch, self._out_channel, kernel_size=1, bias=False))
        self._conv_layers =nn.ModuleList(modules)

        self._conv_final = nn.Conv2d(self._out_channel, 2, kernel_size=1, bias=False)
        
        # 转置卷积上采用
        self._deconv = nn.ConvTranspose2d(self._out_channel, self._out_channel, 
            kernel_size=4, stride=2, padding=1,bias=False)

        self._deconv_final = nn.ConvTranspose2d(self._out_channel, self._out_channel, 
            kernel_size=16, stride=decode_last_stride, padding=4, bias=False)

    def forward(self, encode_data):
        ret = {}
        # encoder中pool5的维度[bs, 512, h/32, w/32]
        input_tensor = encode_data[self._decode_layers[0]]
        # 输入是512维，输出是[bs, 64, h/16, w/16]
        score = self._conv_layers[0](input_tensor)
        # 对于pool4和pool3
        # pool4输出是[bs, 512, h/16, w/16]
        # pool3输出是[bs, 256, h/8, w/8]
        for i, layer in enumerate(self._decode_layers[1:]):
            # 对之前的score进行上采样，得到[bs, 64, h/16, w/16]，之后是[bs, 64, h/8, w/8]
            deconv = self._deconv(score)
            # 读取encode的结果，得到[bs, 64, h/16, w/16]，之后是[bs, 64, h/8, w/8]
            input_tensor = encode_data[layer]
            # 新的score是encode结果的卷积，得到[bs, 64, h/16, w/16]，之后是[bs, 64, h/8, w/8]
            score = self._conv_layers[i](input_tensor)
            # 将相同维度的两个张量叠加，得到[bs, 64, h/16, w/16]，之后是[bs, 64, h/8, w/8]
            fused = torch.add(deconv, score)
            score = fused
        # 将[bs, 64, h/8, w/8]的特征上采用至[bs, 64, h, w]
        deconv_final = self._deconv_final(score)
        # 最后在进行一次卷积，输出[bs, 2, h, w]
        score_final = self._conv_final(deconv_final)

        ret['logits'] = score_final
        ret['deconv'] = deconv_final
        return ret

if __name__ == '__main__':
    torch.cuda.empty_cache()
    from encoders import VGGEncoder
    x = torch.rand((8,3,64,64)).cuda()
    model = VGGEncoder().cuda()
    out = model(x)
    print(out)
    model = FCNDecoder(decode_layers = ["pool5", "pool4", "pool3"], 
        decode_channels=[512,512,256], 
        decode_last_stride=8).cuda()
    # print(model)
    out = model(out)
    for i,j in out.items():
        print(out[i].shape)

    """torch.Size([8, 2, 64, 64])
       torch.Size([8, 64, 64, 64])"""
