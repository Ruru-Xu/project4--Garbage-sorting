# coding=utf-8
# from __future__ import unicode_literals
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from PIL import Image
from dataset.dataloader import read_dataset, garbage_dataloader

# 官方数据集: 训练集和验证集的可视化
def visual_dataset(args, dataset_name='official', figure=False):
    train_img_paths, train_labels, val_img_paths, val_labels, num_img_each_classes = read_dataset(args, dataset_name, split=True)

    # print('set', len(set(train_img_paths)), len(set(train_labels)), len(set(val_img_paths)), len(set(val_labels)))
    len_train = len(train_labels)
    len_val = len(val_labels)
    print('len of dataset: %d' % (len_train+len_val))
    print('len of train_dataset: %d' % (len_train))
    print('len of val_dataset: %d' % (len_val))
    print(('train/val rate: %.2f') % (len_train/len_val))

    train_classes_num = {}
    val_classes_num = {}
    train_val_name_sum_rate = {}

    for label in range(40):
        train_classes_num[label] = 0 # 训练集每个class的数量
        val_classes_num[label] = 0 # 验证集每个class的数量
        train_val_name_sum_rate[label] = {} # 训练/测试的数量比

    for label in train_labels:
        train_classes_num[label] += 1

    for label in val_labels:
        val_classes_num[label] += 1

    for label in range(40):
        name = args.label_id_name_dict[str(label)].split('/')[1]
        train_num = train_classes_num[label]
        val_num = val_classes_num[label]
        sum = train_num + val_num
        rate = round(train_num/val_num, 2)
        train_val_name_sum_rate[label] = {'name':name, 'sum':sum, 'rate':rate}

    ids = list(range(40))
    names = []
    nums = []
    rates = []
    sums = 0

    print(('%-2s, %-8s, %-3s, %-3s') % ('id', 'name', 'num', 'rate' ))
    for i, item in enumerate(train_val_name_sum_rate.values()):
        name, num, rate = item.values()
        names.append(name)
        nums.append(num)
        rates.append(rate)
        sums += num
        print(('%d, %-8s, %d, %.2f') % (i, name, num, rate))

    if figure == True and os.name == 'nt':
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure(1,figsize=(18,9))
        colors = np.random.randn(40) # 随机生成40个颜色
        # print(colors)
        areas = [num/sums*1000 for num in nums]
        plt.scatter(ids, nums,s=areas, c=colors)
        for i, id in enumerate(ids):
            plt.annotate(str(id)+'\n'+str(nums[i])+'\n'+str(rates[i])+'\n'+names[i], (ids[i],nums[i]))
        plt.title('id_name')
        plt.xlabel('id')
        plt.ylabel('num')
        plt.xticks(range(40))
        plt.tight_layout()
        plt.title(['id','num','rate','name'])
        plt.show()

    return ids, names, nums # each ：id, names，nums
