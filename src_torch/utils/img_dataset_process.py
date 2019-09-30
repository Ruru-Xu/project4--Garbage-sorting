# coding=utf-8
'''
    函数 split_classes_data的作用是将数据集划分到40个不同的文件夹中，便于观察分析
'''
from glob import glob
import codecs
import os


# 官方提供的数据集
# 将 train_data 数据集划分到40个文件夹中，存放到 offical_train_data 文件夹中
def split_classes_dataset():
    dataset_path = '../../garbage_classify/train_data'
    save_dataset_path = '../../garbage_classify/official_train_data'
    # read all the img info
    label_files = glob(os.path.join(dataset_path, '*.txt'))
    # make 40 empty dirs
    for i in range(40):
        if not os.path.exists(os.path.join(save_dataset_path, str(i))):
            os.makedirs(os.path.join(save_dataset_path, str(i)))

    # rename each img to new dir
    for file_path in label_files:
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('{} contain error label'.format(os.path.basename(file_path)))
            continue
        img_name = line_split[0]
        img_dir = line_split[1]

        old = os.path.join(dataset_path, img_name)
        new = os.path.join(save_dataset_path,img_dir, img_name)
        os.rename(old, new)

# 对爬取筛选后的数据集 additional_train_data 文件夹和图片名字处理， 文件夹名字一次为0-39, 图片名字为 0-len
# 对文件夹重命名
def rename_dirs():
    root_path = '/home/imc/XR/temp/game/lajifenlei/garbage_classify/garbage_classify/additional_train_data'
    dir_list = os.listdir(root_path)
    print(dir_list)
    for dir in dir_list:
        id_dir = dir.split('_')[0]
        old_dir = os.path.join(root_path, dir)
        new_dir = os.path.join(root_path, id_dir)
        os.rename(old_dir, new_dir)

# 对 additional_train-data 文件夹中的图片重命名
def rename_additional_imgs():
    root_path = '/home/imc/XR/temp/game/lajifenlei/garbage_classify/garbage_classify/additional_train_data'
    dir_list = os.listdir(root_path) # 只是文件名，不是完整的路径
    print(dir_list)
    for dir in dir_list:
        img_list = os.listdir(os.path.join(root_path, dir))
        print(dir, len(img_list))
        for idx, img_name in enumerate(img_list):
            old_path = os.path.join(root_path, dir, img_name)
            new_path = os.path.join(root_path, dir+'_' +dir, 'img_' + str(idx) + '.jpg')
            os.renames(old_path, new_path)
    rename_dirs()

# spider_train-data -->> additional_train_data,
# 爬取数据进行筛选后，将图片移动到 additional_train_data中, 第一张图片的名字为 对应文件夹图片的数量
def move_spider_2_additional():
    spider_path = '../../garbage_classify/spider_train_data'
    additional_path = '../../garbage_classify/additional_train_data'
    spider_dirs = os.listdir(spider_path)
    for spider_dir in spider_dirs: # spider_train_data 文件夹
        label_dir = spider_dir.split('_')[0]  # 文件夹的label
        spider_dir_path = os.path.join(spider_path, spider_dir) # 源文件夹的路径
        additional_dir_path = os.path.join(additional_path, label_dir) # 目标文件夹的路径
        num_additional_dir_img = len(os.listdir(additional_dir_path)) # 目标文件夹中的img数量
        print(label_dir, num_additional_dir_img)
        for img_name in os.listdir(spider_dir_path): # 遍历源文件夹下的图片
            old_name = os.path.join(spider_dir_path, img_name) # 原路径
            new_name = os.path.join(additional_dir_path, str(num_additional_dir_img)+'.jpg') # 新路径
            os.renames(old_name, new_name) # 移动
            num_additional_dir_img += 1 # 目标文件夹中的img数量 +1

if __name__ == '__main__':
    #split_classes_dataset()
    rename_additional_imgs() # 先对additional_train_data的文件夹和图片名字整理, 防止spider_train_data的图片移动时发生 图片已存在,命名重复的问题.
    # move_spider_2_additional() # 将爬取的图片移动到对应的文件夹中, 名字接着目的文件夹的 图片名字

