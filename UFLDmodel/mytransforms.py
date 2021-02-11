import random
import numpy as np
from PIL import Image
import torch


# ===============================img tranforms============================

class Compose2(object):

    def __init__(self, transforms):
        # 输入transforms是一个包括所有变换的列表
        # 对于所有的transforms,对图片和标签逐一进行变换
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class FreeScale(object):
    def __init__(self, size):
        # 输入size是(h, w)
        self.size = size

    def __call__(self, img, mask):
        # 图片用双线性插值，标签用最近邻插值
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]), Image.NEAREST)

class FreeScaleMask(object):
    def __init__(self,size):
        # 输入size是(h, w)
        self.size = size

    def __call__(self,mask):
        # 仅针对mask进行resize
        return mask.resize((self.size[1], self.size[0]), Image.NEAREST)

class Scale(object):
    def __init__(self, size):
        # 输入size是单个的数值，表示缩放后短边的值
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)
        assert img.size == mask.size
        w, h = img.size
        # 如果size等于图片中小的那个尺寸，直接返回
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            # w<h时，小的那个值即w等于self.size，oh等于缩放的尺寸
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomRotate(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, angle):
        # 输入angle是角度变话的范围
        self.angle = angle

    def __call__(self, image, label):
        assert label is None or image.size == label.size

        # 在angle范围内随机旋转
        angle = random.randint(0, self.angle * 2) - self.angle

        label = label.rotate(angle, resample=Image.NEAREST)
        image = image.rotate(angle, resample=Image.BILINEAR)

        return image, label

# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        # 输入均值和方差
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 反标准化
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class RandomLROffsetLABEL(object):
    def __init__(self,max_offset):
        # 输入max_offset是最大的偏移值
        self.max_offset = max_offset

    def __call__(self,img,label):
        # max_offset是个范围，在这个范围内随机左右平移
        offset = np.random.randint(-self.max_offset,self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            # 正向偏移，从偏移值往后的所有列等于从0开始到w-offset的值
            # 然后0到偏移值的位置补充0
            img[:,offset:,:] = img[:,0:w-offset,:]
            img[:,:offset,:] = 0
        if offset < 0:
            real_offset = -offset
            img[:,0:w-real_offset,:] = img[:,real_offset:,:]
            img[:,w-real_offset:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[:,offset:] = label[:,0:w-offset]
            label[:,:offset] = 0
        if offset < 0:
            offset = -offset
            label[:,0:w-offset] = label[:,offset:]
            label[:,w-offset:] = 0
        return Image.fromarray(img),Image.fromarray(label)

class RandomUDoffsetLABEL(object):
    def __init__(self,max_offset):
        # 输入max_offset是最大的偏移值
        self.max_offset = max_offset

    def __call__(self,img,label):
        # max_offset是个范围，在这个范围内随机上下平移
        offset = np.random.randint(-self.max_offset,self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[offset:,:,:] = img[0:h-offset,:,:]
            img[:offset,:,:] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h-real_offset,:,:] = img[real_offset:,:,:]
            img[h-real_offset:,:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:,:] = label[0:h-offset,:]
            label[:offset,:] = 0
        if offset < 0:
            offset = -offset
            label[0:h-offset,:] = label[offset:,:]
            label[h-offset:,:] = 0
        return Image.fromarray(img),Image.fromarray(label)
