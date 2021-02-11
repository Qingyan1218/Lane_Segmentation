import torch, os

import torchvision.transforms as transforms
import mytransforms as mytransforms
from constant import tusimple_row_anchor, culane_row_anchor
from dataset import LaneClsDataset, LaneTestDataset

def get_train_loader(batch_size, data_root, griding_num, dataset, use_aux, distributed):
    """
    Args:
        batch_size: 批次大小
        data_root: 训练图片根目录
        griding_num: 栅格的尺寸
        dataset: 数据集处理的类
        use_aux: 是否使用辅助训练
        distributed: 是否采用分布式

    Returns:
        train_loader: 训练数据加载器
        cls_num_per_lane: 每个车道线的分类数目

    """
    # target_transform 未使用
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])

    # segment_transform 用于label
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])

    # img_transform 用于img
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # simu_transform同时用于img和label
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])
    if dataset == 'CULane':
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'list/train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           segment_transform=segment_transform, 
                                           row_anchor = culane_row_anchor,
                                           griding_num=griding_num, use_aux=use_aux)
        cls_num_per_lane = 18

    elif dataset == 'Tusimple':
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           griding_num=griding_num, 
                                           row_anchor = tusimple_row_anchor,
                                           segment_transform=segment_transform,use_aux=use_aux)
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=0)

    return train_loader, cls_num_per_lane

def get_test_loader(batch_size, data_root,dataset, distributed):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'list/test.txt'),img_transform = img_transforms)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'test.txt'), img_transform = img_transforms)
        cls_num_per_lane = 56

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle = False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler, num_workers=0)
    return loader


class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
     将DistributedSampler的行为更改为顺序分布式采样。
     顺序采样有助于提高多线程测试的稳定性，该测试需要多线程文件io。
     如果不顺序采样，线程上的文件io可能会干扰其他线程。
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size


        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)

if __name__=='__main__':
    import random
    import matplotlib.pyplot as plt
    import numpy as np

    data_root = 'E:/tusimple_0531'
    dataset = 'Tusimple'
    batch_size = 32
    griding_num = 100
    use_aux = True
    distributed = False
    train_loader, cls_num = get_train_loader(batch_size, data_root, griding_num, dataset, use_aux, distributed)
    idx = random.randint(0,10)
    sample = train_loader.dataset[idx]
    # sample[0]是resize后的图像
    print('img.shape',sample[0].shape)
    plt.imshow(sample[0].numpy().transpose(1,2,0))
    plt.title('image')
    plt.show()
    label = sample[1]
    print('label.shape',sample[1].shape)
    # label是一个56行4列的列表，每一行表示4条车道所在的x坐标，
    # 如果值是100，代表没有车道线
    img_label = np.zeros((56,100),dtype = np.uint8)
    for i,row in enumerate(label):
        for j in range(4):
            if row[j]==100:
                continue
            else:
                # 将[row[j]即x处的值变成1
                img_label[i][row[j]] = 1

    plt.imshow(img_label, cmap = 'gray')
    plt.title('binary_lane')
    plt.show()
    # seg_label的形状是36x100
    seg_label = sample[2]
    print('seg_label',seg_label.shape)
    plt.imshow(seg_label)
    plt.title('seg_lane')
    plt.show()
