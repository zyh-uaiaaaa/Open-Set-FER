# -*- coding: utf-8 -*-

import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F

from utils import *
from resnet import *
from torch.autograd import Variable
import torch
import torch.nn as nn

open_num = 0

class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        df = pd.read_csv(args.label_path, sep=' ', header=None)

        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            dataset = df[df[name_c].str.startswith('test')]

        
        
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        
        #### create open set: open_num is the class number indicating open class     
        openidx = []
        for i in range(len(self.label)):
            if self.label[i] != open_num:
                openidx.append(i)
        self.label = np.array(self.label)[np.array(openidx)]
        new_label = []
        for j in range(len(openidx)):
            if self.label[j] < open_num:
                new_label.append(self.label[j])
            else:
                new_label.append(self.label[j]-1)
                
        self.label = np.array(new_label)
        images_names = np.array(images_names)[np.array(openidx)]
        #### end creating open set
        
        self.aug_func = [flip_image, add_g]
        self.file_paths = []

        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)
            

    def __len__(self):
        return len(self.file_paths)
    
    def get_labels(self):
        return self.label
    
    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])
        image = image[:, :, ::-1]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx
    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

class res18feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=6, drop_rate=0.4, out_dim=64):
        super(res18feature, self).__init__()
        #'affectnet_baseline/resnet18_msceleb.pth'
        res18 = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000)
        
        msceleb_model = torch.load('./checkpoint/resnet18_msceleb.pth')
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict, strict=False)
        self.out_dim = out_dim
        self.features = nn.Sequential(*list(res18.children())[:-2])
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])
        self.fc = nn.Linear(args.out_dimension, 6)
        
        self.parm={}
        for name,parameters in self.fc.named_parameters():
            self.parm[name]=parameters
            
    def forward(self, x, target, phase='train'):
        x = self.features(x)        
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='../data/raf-basic',help='raf_dataset_path')
parser.add_argument('--pretrained_backbone_path', type=str, default='./checkpoint/resnet18_msceleb.pth', help='pretrained_backbone_path')
parser.add_argument('--label_path', type=str, default='../data/raf-basic/EmoLabel/list_patition_label.txt', help='label_path')
parser.add_argument('--workers', type=int, default=8, help='number of workers')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--out_dimension', type=int, default=512, help='feature dimension')
args = parser.parse_args()




def train():
    setup_seed(0)
    res18 = res18feature(args)
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RafDataset(args, phase='train', transform=data_transforms)
    test_dataset = RafDataset(args, phase='test', transform=data_transforms_val)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    res18.cuda()
    res18 = torch.nn.DataParallel(res18)
    params = res18.parameters()


    optimizer = torch.optim.Adam([{'params': params}], lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    
    best_acc = 0
    best_epoch = 0
    for i in range(1, args.epochs + 1):
        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        res18.train()

        for batch_i, (imgs, labels, indexes) in enumerate(train_loader):
            imgs = imgs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()            
            outputs = res18(imgs, labels, phase='train')           
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels) 
            
            loss.backward()
            optimizer.step()
            

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            running_loss += loss

        scheduler.step()        
        running_loss = running_loss / iter_cnt
        acc = correct_sum.float() / float(train_dataset.__len__())
        print('Epoch : %d, train_acc : %.4f, train_loss: %.4f' % (i, acc, running_loss))
 
        with torch.no_grad():
            res18.eval()
            running_loss = 0.0
            iter_cnt = 0
            correct_sum = 0
            data_num = 0


            for batch_i, (imgs, labels, indexes) in enumerate(test_loader):
                imgs = imgs.cuda()
                labels = labels.cuda()

                outputs = res18(imgs, labels, phase='test')
                loss = nn.CrossEntropyLoss()(outputs, labels)

                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)

                correct_num = torch.eq(predicts, labels).sum()
                correct_sum += correct_num

                running_loss += loss
                data_num += outputs.size(0)

            running_loss = running_loss / iter_cnt
            test_acc = correct_sum.float() / float(data_num)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = i

            torch.save({'model_state_dict': res18.module.state_dict()}, "0.pth")



            print('Epoch : %d, test_acc : %.4f, test_loss: %.4f' % (i, test_acc, running_loss))

    print('best acc: ', best_acc, 'best epoch: ', best_epoch)


if __name__ == '__main__':
    train()
