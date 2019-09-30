import os
import argparse
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn

from collections import OrderedDict
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataset import custom_transforms as tr
#from models.resnet import resnext101_32x8d
from models.hubconf import resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x8d_wsl

parse = argparse.ArgumentParser(description='the garbar classify competition')

parse.add_argument('--model_name', type=str, default='resnext101_32x16d_wsl')
parse.add_argument('--model_path', type=str, default='/home/imc/XR/temp/game/yunzhuangshibie/pth/resnext101_32x16d_wsl/CrossEntropyLoss_sgd_88_0.0001_5_no_setting/28-9992-0.00080-9902-0.02163.pth')
parse.add_argument('--num_classes', type=int, default=9)
parse.add_argument('--img_size', type=int, default=224)
parse.add_argument('--pretrained', type=bool, default=False)
parse.add_argument('--dropout', type=bool, default=False)
parse.add_argument('--dropout_p', type=float, default=0)

parse.add_argument('--test_img_path', type=str, default='/home/imc/XR/temp/game/yunzhuangshibie/garbage_classify/test')
parse.add_argument('--save_csv_path', type=str, default='/home/imc/XR/temp/game/yunzhuangshibie/outputCSV/submit-0917-16d.csv')

args = parse.parse_args()

def create_model(args):
    model = None
    if args.model_name == 'resnext101_32x16d_wsl':
        model = resnext101_32x16d_wsl(pretrained=args.pretrained, progress=True)
        for param in model.parameters():
            param.requires_grad = False
        if not args.dropout:
            model.fc = nn.Linear(2048, args.num_classes)
        else:
            model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout_p),  # Dropout
                #nn.Linear(2048, 40)
                nn.Linear(2048, args.num_classes)
            )

    if args.model_path:
        #model.load_state_dict(torch.load(args.model_path))
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.model_path).items()})
    return model


class Weather_DataSet(data.Dataset):
    def __init__(self, test_img_path, img_size):
        self.img_list = [os.path.join(test_img_path, img_file) for img_file in os.listdir(test_img_path)]
        self.img_size = img_size
        self.transform = transforms.Compose(
            [
                tr.MaxResize(self.img_size, mode='test'),
                # tr.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),  # 标准化
                tr.ToTensor(),  # 2 tensor
            ]
        )

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        img = self.transform({'image': img})
        return img, os.path.split(self.img_list[index])[1]

    def __len__(self):
        return len(self.img_list)

model = create_model(args)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
test_dataloader = data.DataLoader(Weather_DataSet(args.test_img_path, args.img_size), batch_size=256)

results = OrderedDict([
    ('FileName', []),
    ('type', [])
])
with torch.no_grad():
    for imgs, img_filenames in tqdm(test_dataloader):
        imgs = imgs['image']
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        output = model(imgs)
        _, predicted = torch.max(output.data, 1)   # 预测类别
        predicted +=1
        results['FileName'].extend(list(img_filenames))
        results['type'].extend(predicted.tolist())
        #results['FileName'] = list(img_filenames)
        #results['type'] = predicted.tolist()

df = pd.DataFrame({'FileName': results['FileName'], 'type': results['type']})
df.to_csv(args.save_csv_path, index=False)
