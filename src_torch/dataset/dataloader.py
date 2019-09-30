# coding=utf-8
import os
import random
import codecs
from glob import glob

import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from dataset import custom_transforms as tr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

class garbage_dataset(data.Dataset):
    def __init__(self, args, img_paths, labels, mode=None):
        assert mode == 'train' or mode == 'val'
        self.args = args
        self.img_size = args.input_size
        self.img_paths = img_paths
        self.labels= labels
        self.mode = mode
        self.transform = None
        self.interval_rate = 0.1

        # 在 custom_service.py 文件中一定要记得相应的更改
        if self.mode == 'train':
            self.transform = transforms.Compose(
                [
                    # tr.CenterCrop(self.img_size, self.interval_rate), # 0.1, random = 0.5
                    tr.MaxResize(self.img_size, mode='train'),  # 随机切割, rate=1.2
                    tr.RandomHorizontalFlip(), # 随机水平翻转 0.5
                    # tr.RandomGaussianBlur(), # 随机高斯模糊 # 0.5 random.random()
                    # tr.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), # 标准化 mean= (0.485,0.456,0.406),std= (0.229,0.224,0.225)
                    tr.ToTensor(), # 2 tensor
                ]
            )
        elif self.mode == 'val':
            self.transform = transforms.Compose(
                [
                    tr.MaxResize(self.img_size, mode='val'),
                    # tr.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),  # 标准化
                    tr.ToTensor(),  # 2 tensor
                ]
            )

    # 获取数据集长度
    def __len__(self):
        return len(self.labels)

    # 获取单一数据
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        sample = {'image':img, 'label':label}
        return self.transform(sample)


# 数据集分割
def split_dataset(args, img_paths, labels, way=1):
    train_img_paths = []
    train_labels = []
    val_img_paths = []
    val_labels = []

    # Stratified
    if way == 1:
        skf = StratifiedKFold(n_splits=args.n_splits, random_state=7, shuffle=True)
        for train_index, val_index in skf.split(img_paths, labels):
            train_img_paths, train_labels = list(np.array(img_paths)[train_index]), list(np.array(labels)[train_index])
            val_img_paths, val_labels = list(np.array(img_paths)[val_index]), list(np.array(labels)[val_index])
            break

    # train_test_split
    elif way == 2:
        # labels = np_utils.to_categorical(labels, num_classes) # ong-hot
        train_img_paths, val_img_paths, train_labels, val_labels = \
                train_test_split(img_paths, labels, test_size=1/args.n_splits, random_state=7)

    return train_img_paths, train_labels, val_img_paths, val_labels


# 官方数据集
# read the official dataset paths and split it to train_dataset and val_dataset
def read_dataset(args, dataset_name, split=None, way=1): # 数据集名字, 是否分割, 分割方式
    # label_files = glob(os.path.join(args.dataset_path, '*.txt'))
    # random.shuffle(label_files)
    dataset_path = None
    if dataset_name == 'official':
        dataset_path = args.official_dataset_path
    elif dataset_name == 'additional':
        dataset_path = args.additional_dataset_path

    label_dirs = os.listdir(dataset_path) # 每个label的文件夹名字
    img_paths = [] # 所有图片路径
    labels = [] # 所有label值
    num_img_each_classes = [0]*40 # 每个类别的图片数量


    # 读取数据集
    for dir_name in label_dirs:
        label = int(dir_name)
        dir_path = os.path.join(dataset_path, dir_name)
        num_img_each_classes[label] = len(os.listdir(dir_path))
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img_paths.append(img_path)
            labels.append(label)

    if split == True:
        train_img_paths, train_labels, val_img_paths, val_labels = split_dataset(args, img_paths, labels, way)
        return train_img_paths, train_labels, val_img_paths, val_labels, np.array(num_img_each_classes)
    else:
        return img_paths, labels, np.array(num_img_each_classes)


def garbage_dataloader(args):
    train_img_paths, train_labels, train_num_img_each_classes, val_img_paths, val_labels, val_num_img_each_classes = None, None, None, None, None, None
    if args.val == True: # val集可以不要,但是一般还是要的
        train_img_paths, train_labels, val_img_paths, val_labels, num_img_each_classes = read_dataset(args, dataset_name='official', split=True)
        train_num_img_each_classes = num_img_each_classes / args.n_splits * (args.n_splits - 1)
        val_num_img_each_classes = num_img_each_classes / args.n_splits
    else:
        train_img_paths, train_labels, num_img_each_classes = read_dataset(args, dataset_name='official', split=False)
        train_num_img_each_classes = num_img_each_classes



    if args.use_additional_dataset == True: # 是否使用additional_train_dataset
        if args.split_additional_dataset == False or args.val == False: # 是否将 additional_train_dataset 数据集分成 train 和 test 两部分
            additional_img_paths, additional_labels, additional_num_img_each_classes = read_dataset(args, dataset_name='additional', split=False)
            train_img_paths.extend(additional_img_paths)
            train_labels.extend(additional_labels)
            train_num_img_each_classes = train_num_img_each_classes + additional_num_img_each_classes
        # else: # 扩充的数据集一般是不会用来分成 train and val两个训练集的
        #     additional_train_img_paths, additional_train_labels, additional_val_img_paths, additional_val_labels, additional_num_img_each_classes \
        #         = read_dataset(args, dataset_name='additional', split=True)
        #     train_img_paths.extend(additional_train_img_paths)
        #     train_labels.extend(additional_train_labels)
        #     val_img_paths.extend(additional_val_img_paths)
        #     val_labels.extend(additional_val_labels)


    train_dataset = garbage_dataset(args, train_img_paths, train_labels, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_worker, shuffle=True)
    if args.val == True:
        val_dataset = garbage_dataset(args, val_img_paths, val_labels, mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_worker, shuffle=False)
        return train_dataloader, train_num_img_each_classes, val_dataloader, val_num_img_each_classes
    else:
        return train_dataloader, train_num_img_each_classes

if __name__ == '__main__':
    pass







