import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


class LaneDataSet(Dataset):
    def __init__(self, dataset_file, transform=None):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform

        with open(dataset_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                self._gt_label_instance_list.append(info_tmp[2])


    def _split_instance_gt(self, label_instance_img):
        # 实例分割用的图像里面有5种颜色，而且不是从0~4，所以需要变成one_hot
        # 但是图像数据并非都有5种颜色，少的只有3种
        num_of_instances = 5
        # 创建一个数组，第一维是channel
        ins = np.zeros((num_of_instances, label_instance_img.shape[0],label_instance_img.shape[1]))
        # np.unique取出颜色的几个数值，因为0已经存在，所以从1开始
        # 通道0,1,2,3匹配四种颜色的列表，第i个通道让所有等于第i颜色的值变为1，余同
        # 因此对于少于5种颜色的图像来说，剩下的通道都是0
        for _ch, label in enumerate(np.unique(label_instance_img)[1:]):
            ins[_ch, label_instance_img == label] = 1
        return ins

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        # 用PIL.Image读取图片方便和torch搭配使用，原始图片和label_img要按三通道读取
        img = Image.open(self._gt_img_list[idx]).convert('RGB')
        label_instance_img = Image.open(self._gt_label_instance_list[idx])
        label_img = Image.open(self._gt_label_binary_list[idx]).convert('RGB')

        # 图像增广，其中为了不改变颜色的数值，都用最近邻插值
        if self.transform:
            img = self.transform(img)
        C,H,W = img.shape
        label_transforms = transforms.Resize((H,W),interpolation = 0)
        label_img = np.array(label_transforms(label_img))[46:]
        label_instance_img = np.array(label_transforms(label_instance_img))[46:]

        label_instance_img = self._split_instance_gt(label_instance_img)

        # img = img.transpose(2,0,1)
        # label_img是按三通道打开的，因此不是全0的地方就表示前景
        # 也不能按照灰度图打开，因为非二值图像
        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1
        img = torch.FloatTensor(img[:,46:,])
        label_binary = torch.LongTensor(label_binary)
        label_instance_img = torch.FloatTensor(label_instance_img)
        # print(img.shape,label_binary.shape,label_instance_img.shape)

        return img, label_binary, label_instance_img

if __name__ == '__main__':
    dataset_file = r'..\tusimple_0531\training\train_for_test.txt'
    train_dataset = LaneDataSet(dataset_file,transform=transforms.Compose([transforms.Resize((256,480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35248518, 0.37250566, 0.3955756],
                std=[0.20043196, 0.2173182, 0.2510542])]))
    train_loader = DataLoader(dataset = train_dataset , batch_size = 2, shuffle = True)

    for batch in train_loader:
        """ ret = {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': decode_logits}"""
        road_gt = batch[0]
        image_data = road_gt
        seg_gt = batch[1].numpy()
        # print(seg_gt.shape) # (4, 224, 480)
        instance_input = batch[2].numpy()
        # print(instance_input.shape) # (4, 5, 224, 480)
        instance_gt = np.argmax(instance_input, axis=1).astype('uint8')

        for s_gt, i_gt, r_gt in zip(seg_gt, instance_gt, road_gt):
            print('-' * 60)
            plt.figure()
            plt.subplot(231)
            plt.title('seg_gt')
            plt.imshow(s_gt, cmap='gray')

            plt.subplot(232)
            plt.title('instance_gt')
            plt.imshow(i_gt)

            plt.subplot(233)
            plt.title('ground_truth')
            plt.imshow(r_gt.numpy().transpose(1, 2, 0))
            plt.show()

        break



    
