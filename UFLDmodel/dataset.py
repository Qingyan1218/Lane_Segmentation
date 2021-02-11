import torch
from PIL import Image
import os
import pdb
import numpy as np


def loader_func(path):
    return Image.open(path)

def find_start_pos(row_sample,start_line):
    # row_sample 是所有y值列表，有56个值
    # start_line 是最后一个点的y值
    # l(left)从0开始，r(right)从55开始
    l,r = 0,len(row_sample)-1
    while True:
        # 通过夹逼查找start_line（最后一个y值）在整个56行中的位置
        # 取l和r中间的y
        mid = int((l+r)/2)
        if r - l == 1:
            return r
        # 如果row_sample中间位置的y比开始位置小，则left=mid
        if row_sample[mid] < start_line:
            l = mid
        # 如果row_sample中间位置的y比开始位置大，则right=mid
        if row_sample[mid] > start_line:
            r = mid
        if row_sample[mid] == start_line:
            return mid



class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]


    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        # 打开图片
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform = None,target_transform = None,simu_transform = None, griding_num=100, load_name = False,
                row_anchor = None,use_aux=False,segment_transform=None, num_lanes = 4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        # 是否要加载图片的名称
        self.load_name = load_name
        # use_aux是训练时是否使用辅助
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        # 打开图片和标签
        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        # 在row anchors处获得车道线及延长的车道线的坐标，4x56x[y,x]
        lane_pts = self._get_index(label)


        w, h = img.size
        # 将坐标变成分类标签，即56x4，值为车道线的x坐标
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)

        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            # img.shape torch.Size([3, 288, 800])
            # label.shape (56, 4)
            # seg_label torch.Size([36, 100])
            return img, cls_label, seg_label


        if self.load_name:
            return img, cls_label, img_name


        return img, cls_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts是4x56x[y,x]的矩阵，num_cols=100
        num_lane, n, n2 = pts.shape
        # 0到w-1分成100份，这样x坐标最大是99，不影像用100表示无车道线
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        # to_pts:56x4
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            # 取出每根车道线对应的所有x坐标
            pti = pts[i, :, 1]
            # 对于pti中的pt，如果pt！=-1，x的值为int(pt // (col_sample[1] - col_sample[0]))
            # 否则x的值就等于100，to_pts相应车道线位置的值变成x坐标
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size

        if h != 288:
            # 图片还没有transform，因此h都不等于288，因此总是有sample_tmp
            # 计算缩放系数
            scale_f = lambda x : int((x * 1.0/288) * h)
            # 将所有的anchor坐标放大到原图的尺寸
            sample_tmp = list(map(scale_f,self.row_anchor))
        # 建立一个4x56x2的矩阵，4x56x[y,x]
        all_idx = np.zeros((self.num_lanes,len(sample_tmp),2))
        for i,r in enumerate(sample_tmp):
            # r进行取整，label_r.shape = (1280,),label非矩阵，形状是720x1280
            # np.asarray对label进行原位修改，label_r即label的第r行
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                # 将label_r中等于车道线索引的x坐标取出来
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    # 如果该行没有车道线，all_idx中对应于车道线所在的列
                    # 第一个值设为r，第二个值设为-1
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                # 否则位置取平均值，all_idx中对应于车道线所在的列
                # 第一个值设为人，第二个值设为x坐标
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # 数据增强：将车道线延伸到边界
        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            # 如果all_idx_cp的最后一维的第二个值是-1，代表没有车道线
            if np.all(all_idx_cp[i,:,1] == -1):
                continue
            # 获得有效车道线的x坐标，即不等于-1的列
            valid = all_idx_cp[i,:,1] != -1
            # 获得某根车道中有效车道线的坐标[[y,x],[y,x],[y,x]...]
            valid_idx = all_idx_cp[i,valid,:]
            # 如果有效车道线的最后一个y值和所有行的最后一行的y值，意味着车道线达到了底部，跳过
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:
                continue
            # 如果车道线太短，舍弃
            if len(valid_idx) < 6:
                continue

            # 车道线下半部分的值
            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            # 用y拟合x
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
            # start_line是原车道线的最后一个点的y值
            start_line = valid_idx_half[-1,0]
            # 查找最后一个y值在整个列表（56行）中的位置
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            # 用直线方程求出pos后的所有x坐标
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            # 将超出图像范围的值去掉
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])
            # 确保延伸的线所在的地方是没有车道线的，否则没有延伸的必要
            assert np.all(all_idx_cp[i,pos:,1] == -1)
            # 将延长的车道线写入原来的数据
            all_idx_cp[i,pos:,1] = fitted

        # if -1 in all_idx[:, :, 0]:
        #     pdb.set_trace()
        # 返回一个4x56x2的矩阵，是在原图基础上的
        return all_idx_cp

if __name__=='__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import random
    tusimple_row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                           116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                           168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                           220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                           272, 276, 280, 284]

    data_root = 'E:/tusimple_0531'

    train_dataset = LaneClsDataset(data_root,
                                   os.path.join(data_root, 'train_gt.txt'),
                                   row_anchor = tusimple_row_anchor,
                                   use_aux=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    plt.imshow(sample[0])
    plt.show()
    label = sample[1]
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
    plt.show()
