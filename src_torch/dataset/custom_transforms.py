import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter


# 先等比例crop再resize
class CenterCrop(object):
    def __init__(self, crop_size, interval_rate):
        self.crop_size = crop_size
        self.interval_rate = interval_rate

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        w, h = img.size
        if h/w < 2 and w/h > 2 and random.random() <= 0.5: # 只用 h/w
            w1 = self.interval_rate * w
            w2 = (1-self.interval_rate) * w
            h1 = self.interval_rate * h
            h2 = (1-self.interval_rate) * h
            img = img.crop((w1, h1, w2, h2))
            # img = img.resize((w, h), Image.BILINEAR)
        return {'image': img,
                'label': label}


class MaxResize(object):
    def __init__(self, crop_size, mode):
        self.crop_size= crop_size
        self.rate = 1.2
        self.mode = mode

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        w, h = img.size
        # if h/w >= 1.85: # 高是宽的两倍 2
            # copy 2 same imgs in a img
            # img = self.double_img(img) # 水平翻转
            # full in a 224x224 img center
            # img = self.center_img(img) # 效果不好
        if w/h<=self.rate and h/w<=self.rate:  # 直接resize
            pass
        elif w < h: # 宽小于高
            w = int(w*self.rate)
            if self.mode == 'train':
                y = random.randint(0, int((h-w)/2))
            elif self.mode == 'val':
                y = int((h-w)/2)
            img = img.crop((0, y, w, y+w))
        elif w > h: # 宽大于高
            h = int(h*self.rate)
            if self.mode == 'train':
                x = random.randint(0, int((w-h)/2))
            elif self.mode == 'val':
                x = int((w-h)/2)
            img = img.crop((x, 0, x+h, h))

        img = img.resize((self.crop_size, self.crop_size))
        return {'image':img,
                'label':label}

    def double_img(self, img):
        w, h = img.size
        new_img = np.zeros((h, 2 * w, 3), dtype=np.uint8)
        array_img = np.asarray(img)
        new_img[:, :w, :] = array_img
        new_img[:, w:, :] = np.fliplr(array_img)
        img = Image.fromarray(new_img)
        return img

    def center_img(self, img):
        w, h = img.size
        img = np.asarray(img)
        l = max(w, h)
        new_img = np.full((l, l, 3), 255, dtype=np.uint8)
        if w < h:
            x = int((h-w)/2)
            new_img[:, x:x+w, :] = img
        elif w > h:
            y = int((w-h)/2)
            new_img[y:y+h, :, :] = img
        img = Image.fromarray(new_img)
        return img


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT) #图片左右翻转
        return {'image': img,
                'label': label}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': label}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        #转array最后再由ToTensor转成张量
        img = np.array(img,dtype=np.float64)
        img /= 255
        img -= self.mean
        img /= self.std
        # img = Image.fromarray(img)
        return {'image': img,
                'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        label = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        label = np.array(label).astype(np.float32)

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()

        return {'image': img,
                'label': label}









class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        lbl = sample['label']
        w, h = img.size
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        lbl = lbl.crop((x, y, x + self.crop_size, y + self.crop_size))

        return {'image': img,
                'label': lbl}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        label = label.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': label}

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w) # 按比例取h的值
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        #label的resize使用参考最近点像素插值
        label = label.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0) # 在右方和下方expand, 0表示padding
            label = ImageOps.expand(label, border=(0, 0, padw, padh), fill=self.fill) # cityscapes中fill=255表示忽略，而不是unlabeled
        # random crop crop_size
        w, h = img.size
        # x1为左方裁的值，y1为上方裁的值，左右裁的值不一样，上下裁的值也不一样,裁完之后为crop_size*crop_size的图片
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        label = label.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': label}




class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']

        assert img.size == label.size

        img = img.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': label}

