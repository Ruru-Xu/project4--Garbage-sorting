# coding:utf-8
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# dataset
from dataset.dataloader import garbage_dataloader, read_dataset
# models
from models.resnet import resnet50
from models.hubconf import resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x8d_wsl
from models.efficientNet import EfficientNet as efficientnet
from models.densenet import densenet121, densenet161, densenet169, densenet201
from models.se_resnet import *
from models.cbam_resnet import *
# utils
from utils.visualization import visual_dataset
from utils.FocalLoss import FocalLoss



parse = argparse.ArgumentParser(description='the garbar classify competition')

# img
parse.add_argument('--official_dataset_path', type=str, default='../garbage_classify/train_data/')
parse.add_argument('--additional_dataset_path', type=str, default='../garbage_classify/additional_train_data')
parse.add_argument('--num_classes', type=int, default=40)
parse.add_argument('--num_sample', type=int, default=None)
parse.add_argument('--input_size', type=int, default=224)
parse.add_argument('--label_id_name_dict', type=dict, default={})
parse.add_argument('--n_splits', type=int, default=4)

# train
parse.add_argument('--batch_size', type=int, default=32)
parse.add_argument('--epoch', type=int, default=10)
parse.add_argument('--num_worker', type=int, default=2)
parse.add_argument('--pretrained', type=bool, default=True)
parse.add_argument('--cuda', type=bool, default=True)
parse.add_argument('--gpu_id', type=str, default='0')
parse.add_argument('--val', type=bool, default=True)
parse.add_argument('--dropout', type=bool, default=False)
parse.add_argument('--dropout_p', type=float, default=0.5)
parse.add_argument('--model_name', type=str, default='resnext101_32x32d_wsl')
parse.add_argument('--use_additional_dataset', type=bool, default=None, help='use the additional dataset or not')
parse.add_argument('--split_additional_dataset', type=bool, default=False, help='if use additional dataset, split it or not')

# optimizer
parse.add_argument('--optimizer', type=str, default='sgd')
parse.add_argument('--lr', type=float, default=1e-4)
parse.add_argument('--lr_fc_times', type=int, default=1, help='the lr of fc layer times')
parse.add_argument('--lr_scheduler', type=str, default=None)
parse.add_argument('--momentum', type=float, default=0.9, metavar='M')
parse.add_argument('--weight_decay', type=float, default=0.0) #5e-4
parse.add_argument('--nesterov', type=bool, default=False)

# criterion
parse.add_argument('--criterion_name', type=str, default='CrossEntropyLoss')

# save pth .pth
parse.add_argument('--save_path', type=str, default=None)
parse.add_argument('--save_accuracy', type=float, default=0.90)
parse.add_argument('--deply_script_path', type=str, default='s3://garbage-classify-tim/src_baseline/deploy_scripts',
                   help='a path which contain config.json and customize_service.py,if it is set, these two scripts wil be copied to {train_url}/pth directory')
parse.add_argument('--freeze_weights_file_path', type=str, default='if it is set, the specified h5 weights file will be converted as a .pth pth, only valid when {pth}=save_pth')

args = parse.parse_args()

args.label_id_name_dict = \
        {
            "0": "其他垃圾/一次性快餐盒",
            "1": "其他垃圾/污损塑料",
            "2": "其他垃圾/烟蒂",
            "3": "其他垃圾/牙签",
            "4": "其他垃圾/破碎花盆及碟碗",
            "5": "其他垃圾/竹筷",
            "6": "厨余垃圾/剩饭剩菜",
            "7": "厨余垃圾/大骨头",
            "8": "厨余垃圾/水果果皮",
            "9": "厨余垃圾/水果果肉",
            "10": "厨余垃圾/茶叶渣",
            "11": "厨余垃圾/菜叶菜根",
            "12": "厨余垃圾/蛋壳",
            "13": "厨余垃圾/鱼骨",
            "14": "可回收物/充电宝",
            "15": "可回收物/包",
            "16": "可回收物/化妆品瓶",
            "17": "可回收物/塑料玩具",
            "18": "可回收物/塑料碗盆",
            "19": "可回收物/塑料衣架",
            "20": "可回收物/快递纸袋",
            "21": "可回收物/插头电线",
            "22": "可回收物/旧衣服",
            "23": "可回收物/易拉罐",
            "24": "可回收物/枕头",
            "25": "可回收物/毛绒玩具",
            "26": "可回收物/洗发水瓶",
            "27": "可回收物/玻璃杯",
            "28": "可回收物/皮鞋",
            "29": "可回收物/砧板",
            "30": "可回收物/纸板箱",
            "31": "可回收物/调料瓶",
            "32": "可回收物/酒瓶",
            "33": "可回收物/金属食品罐",
            "34": "可回收物/锅",
            "35": "可回收物/食用油桶",
            "36": "可回收物/饮料瓶",
            "37": "有害垃圾/干电池",
            "38": "有害垃圾/软膏",
            "39": "有害垃圾/过期药物"
        }


def train_model(args):
    # device
    set_device_environ(args)


    # dataloader
    if args.val == True:
        train_dataloader, train_num_img_each_classes, val_dataloader, val_num_img_each_classes = garbage_dataloader(args)
        num_train_batch_size = len(train_dataloader)  # 一个epoch有多少个batch_size
        num_val_batch_size = len(val_dataloader)
        print('train:', num_train_batch_size*args.batch_size, 'val:', num_val_batch_size*args.batch_size)
    else:
        train_dataloader, train_num_img_each_classes = garbage_dataloader(args)
        num_train_batch_size = len(train_dataloader)
        val_dataloader, val_num_img_each_classes = None
        print('train:', num_train_batch_size * args.batch_size)

    # model
    model = create_model(args.model_name, pretrained=args.pretrained)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
    model = torch.nn.DataParallel(model)
    #model = model.cuda()

    # optimizer
    optimizer = create_optimizer(args, model)

    # lr_scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 40], gamma=0.1)

    # criterion
    criterion = create_criterion(criterion_name=args.criterion_name)

    # device
    parallel = False
    if args.cuda:
        if len(args.gpu_id.split(',')) > 1 and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            parallel = True

    print(torch.cuda.is_available())
    # device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() and args.cuda==True  else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda==True  else 'cpu')
    print('device:',device)
    #print(torch.cuda.is_available())
    #model.to(device)

    # training
    for epoch in range(args.epoch):
        model.train()
        loss_sum = 0.0
        tbar = tqdm(train_dataloader, ncols=100)  # 进度条
        for i, sample in enumerate(tbar):
            # get a batch_size sample
            img, label = sample['image'], sample['label']

            img = img.to(device)
            label = label.to(device)

            # optimizer to clean the gradient buffer
            optimizer.zero_grad()

            # forward
            output = model(img)

            # compute the loss
            loss = criterion(output, label)

            # backward
            loss.backward()

            # optimize
            optimizer.step()

            # 每个batch_size 打印 train_loss
            loss_sum += loss.item()
            tbar.set_description('epoch:%d, train loss:%.4f ' % (epoch, loss_sum/(i+1)))


        train_loss, train_accuracy, train_each_class_accuracy = 0.0, 0.0, []
        val_loss, val_accuracy, val_each_class_accuracy = 0.0, 0.0, []

        # train and val the training model
        kwargs = {'model':model, 'criterion':criterion, 'device':device}
        train_loss, train_accuracy, train_each_class_true_label = train_val_model(args, train_dataloader, **kwargs)
        train_each_class_accuracy = np.around(train_each_class_true_label/train_num_img_each_classes, 4)

        if args.val == True:
            val_loss, val_accuracy, val_each_class_true_label = train_val_model(args, val_dataloader, **kwargs)
            val_each_class_accuracy = np.around(val_each_class_true_label/val_num_img_each_classes, 4)
            print(('train_loss: %.4f, train_accuracy: %.4f, val_loss: %.4f, val_accuracy: %.4f\n') %
                  (train_loss, train_accuracy, val_loss, val_accuracy))
        else:
            print(('train_loss: %.4f, train_accuracy: %.4f\n') % (train_loss, train_accuracy))

        # save the train and val confusion matrix, the json file name
        if args.val == True:
            json_name = args.save_path+'/'+str(epoch)+'-'+str(train_accuracy)[2:6]+'-'+str(train_loss)[:7]\
                           +'-'+str(val_accuracy)[2:6]+'-'+str(val_loss)[:7]+'.json'
            pth_name = args.save_path+'/'+str(epoch)+'-'+str(train_accuracy)[2:6]+'-'+str(train_loss)[:7]\
                           +'-'+str(val_accuracy)[2:6]+'-'+str(val_loss)[:7]+'.pth'
        else:
            json_name = args.save_path+'/'+str(epoch)+'-'+str(train_accuracy)[2:6]+'-'+str(train_loss)+'.json'
            pth_name = args.save_path+'/'+str(epoch)+'-'+str(train_accuracy)[2:6]+'-'+str(train_loss)+'.pth'

        # save the dict of each class accuracy in file.json
        train_accuracy_dict, val_accuracy_dict = {}, {}
        for label, (train_acc, train_num_img, val_acc, val_num_img) in enumerate(zip(train_each_class_accuracy, train_num_img_each_classes,
                                                                                     val_each_class_accuracy, val_num_img_each_classes)):
            train_accuracy_dict[label] = str(train_acc)+' -- '+str(int((1-train_acc)*train_num_img))
            val_accuracy_dict[label] = str(val_acc)+' -- '+str(int((1-val_acc)*val_num_img))

        with open(json_name, 'a') as f:
            f.write(str(epoch)+'_train_loss: '+str(train_loss)+'\n')
            f.write(str(epoch)+'_val_loss:   '+str(val_loss)+'\n\n')
            f.write(str(epoch)+'_train_each_class_accuracy\n')
            json.dump(train_accuracy_dict, f, indent=4)
            f.write('\n\n'+str(epoch)+'_val_each_class_accuracy\n')
            json.dump(val_accuracy_dict, f, indent=4)

        if args.val == True:
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        # save model.pth
        if  val_accuracy > args.save_accuracy or (args.val == False and train_accuracy > args.save_accuracy):
            if parallel == True:
                torch.save(model.module.state_dict(), pth_name)
            else:
                torch.save(model.state_dict(), pth_name)


# 训练过程中的model对 train_dataloader 和 val_dataloader 进行验证
def train_val_model(args, dataloader=None, model=None, criterion=None, device=None):
    model.eval()
    loss = 0.0  # 损失值
    true_label = 0  # 准确数
    accuracy = 0.0  # 准确率
    each_class_true_label = [0]*40 # 每个类别的准确率

    with torch.no_grad():
        # 训练集测试
        for sample in dataloader:
            img, label = sample['image'], sample['label']
            img = img.to(device)
            label = label.to(device)

            # 预测分数
            output = model(img)

            # 计算损失值
            batch_size_loss = criterion(output, label)
            loss += batch_size_loss

            # 计算准确率
            _, predicted = torch.max(output.data, 1)  # 预测类别
            true_label += (predicted == label).sum().item()

            for truth, pred in zip(label, predicted):
                if truth == pred:
                    each_class_true_label[truth] += 1
        loss /= len(dataloader)
        accuracy = true_label / (len(dataloader) * args.batch_size)

        return loss.item(), accuracy, np.array(each_class_true_label)


# 创建 optimizer
def create_optimizer(args, model):
    optimizer = None
    # 不同的网络层不同的学习率
    parameters = []
    for name, param in model.named_parameters():
        if 'fc' in name or 'class' in name:
            parameters.append({'params': param, 'lr': args.lr*args.lr_fc_times})
        else:
            parameters.append({'params': param, 'lr': args.lr})

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            parameters,
            # model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            # weight_decay=args.weight_decay,
            nesterov=False,
        )

    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr = args.lr,
        )
    return optimizer


def create_criterion(criterion_name):
    criterion = None
    if criterion_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    elif criterion_name == 'BCELoss':
        criterion = nn.BCELoss(reduction='mean')
    elif criterion_name == 'FocalLoss':
        alpha = [1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5,  # 2, 8, 9
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0,  # 16
                 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0,  # 22, 23, 27
                 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0,] # 31, 32, 33, 36
        criterion = FocalLoss(num_class=40, alpha=alpha, gamma=2)
    return criterion


# 创建网络模型
def create_model(model_name, pretrained):
    model = None
    if model_name == 'resnet50':
        model = resnet50(pretrained=pretrained, progress=True)
        if not args.dropout:
            model.fc = nn.Linear(2048, 40)
        else:
            model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout_p),  # Dropout
                nn.Linear(2048, 40)
            )
        return model
    elif model_name == 'resnext101_32x8d_wsl':
        model = resnext101_32x8d_wsl(pretrained=pretrained, progress=True)
        if not args.dropout:
            model.fc = nn.Linear(2048, 40)
        else:
            model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout_p),  # Dropout
                nn.Linear(2048, 40)
            )
        return model
    elif model_name == 'resnext101_32x16d_wsl':
        model = resnext101_32x16d_wsl(pretrained=pretrained, progress=True)
        if not args.dropout:
            model.fc = nn.Linear(2048, 40)
        else:
            model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout_p),  # Dropout
                nn.Linear(2048, 40)
            )
        return model
    elif model_name == 'resnext101_32x32d_wsl':
        model = resnext101_32x32d_wsl(pretrained=pretrained, progress=True)
        if not args.dropout:
            model.fc = nn.Linear(2048, 40)
        else:
            model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout_p), # Dropout
                nn.Linear(2048, 40)
            )
        return model
    elif model_name == 'densenet121':
        model = densenet121(pretrained=pretrained, progress=True)
        model.classifier = nn.Linear(1024, 40)
        return model
    elif model_name == 'densenet169':
        model = densenet169(pretrained=pretrained, progress=True)
        model.classifier = nn.Linear(1664, 40)
        return model
    elif model_name == 'densenet201':
        model = densenet201(pretrained=pretrained, progress=True)
        model.classifier = nn.Linear(1920, 40)
        return model
    elif model_name == 'efficientnet_b7':
        if pretrained == True:
            model = efficientnet.from_pretrained('efficientnet-b7', num_classes=40)
        elif pretrained == False:
            model = efficientnet.from_name('efficientnet-b7', override_params={'num_classes': 40})
        return model
    elif model_name == 'se_resnet':
        model = se_resnet101(pretrained=False)
        model_pth_path = '../model_pth/se_resnet101.pth.tar'
        checkpoint_state_dict = torch.load(model_pth_path, map_location='cpu')['state_dict']
        for layer_name in model.state_dict():
            checkpoint_state_dict[layer_name] = checkpoint_state_dict['module.'+layer_name]
            del checkpoint_state_dict['module.'+layer_name]
        model.load_state_dict(checkpoint_state_dict)
        model.fc = nn.Linear(2048, 40)
        return model
    elif model_name == 'cbam_resnet':
        model = cbam_resnet101(pretrained=False)
        model_pth_path = '../model_pth/cbam_resnet101.pth.tar'
        checkpoint_state_dict = torch.load(model_pth_path, map_location='cpu')['state_dict']
        for layer_name in model.state_dict():
            checkpoint_state_dict[layer_name] = checkpoint_state_dict['module.'+layer_name]
            del checkpoint_state_dict['module.'+layer_name]
        model.load_state_dict(checkpoint_state_dict)
        model.fc = nn.Linear(2048, 40)

# device
def set_device_environ(args):
    # gpu
    if os.name == 'nt':
        print('system: windows')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
        args.gpu_id = '0, 1, 2, 3'

    elif os.name == 'posix':
        print('system: linux')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
        args.gpu_id = '0, 1, 2, 3'


if __name__ == '__main__':
    # dataset
    args.official_dataset_path = '../garbage_classify/official_train_data'
    args.additional_dataset_path = '../garbage_classify/additional_train_data'
    # train
    args.cuda = True
    args.pretrained = True
    args.val = True
    args.lr = 1e-4
    args.lr_fc_times = 5
    args.batch_size = 6
    args.epoch = 200
    args.n_splits = 5
    args.save_accuracy = 0.92  # 保存pth的最小准确率
    args.criterion_name = 'CrossEntropyLoss' # ['CrossEntropyLoss', 'FocalLoss', 'BCELoss']
    args.optimizer = 'sgd'    # ['sgd', 'adam'] , 一般使用sgd
    args.model_name = 'resnext101_32x32d_wsl'  # ['resnext101_32x32d_wsl', 'se_resnet', 'resnet50', 'resnext101_32x16d_wsl', 'densenet121', 'densenet169', 'densenet201' 'efficientnet_b7']
    args.use_additional_dataset = False  # 使用外部数据
    args.split_additional_dataset = False  # 外部数据集分割（
    args.dropout = False  # 正则化, 防止过拟合
    args.dropout_p = 0.25 # p=0.25

    args.save_path = '../pth/'+args.model_name+'/'+args.criterion_name+'_'+args.optimizer+'_'+str(args.batch_size)+'_'+str(args.lr)+'_'\
                     +str(args.lr_fc_times)+'_no_setting'
    print('model:', args.model_name)
    print('n_split:', args.n_splits)
    print('optimizer:', args.optimizer)
    print('dropout:', args.dropout, ' p:', args.dropout_p)
    print('batch_size:', args.batch_size)
    print('learning_rate:', args.lr  ,' lr_fc_times:', args.lr_fc_times)

    # train_model
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    train_model(args)

    # visual the dataset
    # ids, names, nums = visual_dataset(args, dataset_name='additional', figure=True) # dataset_nam=['official', 'additional']


