
import os
import argparse
import numpy as np
from tqdm import tqdm
import copy
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms
from mobius_transformation import Mobius
from transformation import super_transformation
import torchvision.models as models
import torchvision
from util.misc import CSVLogger
from util.cutout import Cutout
from numpy.random import randint
from combine_datasets import combine_datasets
import xlwt 
from xlwt import Workbook 

from model.resnet import ResNet18, ResNet50
from model.wide_resnet import WideResNet

model_options = ['resnet18', 'wideresnet','resnet50','tiny']
dataset_options = ['cifar10', 'cifar100', 'svhn','imagenet']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar100',
                    choices=dataset_options)

args = parser.parse_args()

if args.dataset == 'cifar10':
    num_classes = 10
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
 
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])
    test_dataset = datasets.CIFAR10(root='data/',
                                train=False,
                                transform=test_transform,
                                download=True)

elif args.dataset == 'cifar100':
    num_classes = 100
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ## normalize and totensor is in super_transformation

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])
    
    test_dataset = datasets.CIFAR100(root='/deep/group/sharonz/cifar100/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)

elif args.dataset == 'tiny':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    num_classes = 200
    traindir = 'tiny-imagenet-200/train'
    valdir = 'tiny-imagenet-200/val/images'
    
    
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])   
    test_dataset = datasets.ImageFolder(valdir,
                                     transform=test_transform)

print('Start loading model')

cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
# cnn.load_state_dict(torch.load(path))
# path = '/sailhome/jiequanz/mobius-cutout/checkpoints/cifar10_wideresnet_fixedmobius_mix2_with_inter_6/best_model_at_epoch_199.pt'
path = '/sailhome/jiequanz/mobius-cutout/new_checkpoints/cifar100_wideresnet_mix3_8mobius_7/cifar100_mobius_mix3.pt'
path = 'tiny_resnet50_mix3_madmissable_3_best_model_at_epoch_97.pt'

state_dict = torch.load(path)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
cnn.load_state_dict(new_state_dict)

cnn = cnn.cuda()

print('Finish loading model')




coarse_label = [
'apple', # id 0
'aquarium_fish',
'baby',
'bear',
'beaver',
'bed',
'bee',
'beetle',
'bicycle',
'bottle',
'bowl',
'boy',
'bridge',
'bus',
'butterfly',
'camel',
'can',
'castle',
'caterpillar',
'cattle',
'chair',
'chimpanzee',
'clock',
'cloud',
'cockroach',
'couch',
'crab',
'crocodile',
'cup',
'dinosaur',
'dolphin',
'elephant',
'flatfish',
'forest',
'fox',
'girl',
'hamster',
'house',
'kangaroo',
'computer_keyboard',
'lamp',
'lawn_mower',
'leopard',
'lion',
'lizard',
'lobster',
'man',
'maple_tree',
'motorcycle',
'mountain',
'mouse',
'mushroom',
'oak_tree',
'orange',
'orchid',
'otter',
'palm_tree',
'pear',
'pickup_truck',
'pine_tree',
'plain',
'plate',
'poppy',
'porcupine',
'possum',
'rabbit',
'raccoon',
'ray',
'road',
'rocket',
'rose',
'sea',
'seal',
'shark',
'shrew',
'skunk',
'skyscraper',
'snail',
'snake',
'spider',
'squirrel',
'streetcar',
'sunflower',
'sweet_pepper',
'table',
'tank',
'telephone',
'television',
'tiger',
'tractor',
'train',
'trout',
'tulip',
'turtle',
'wardrobe',
'whale',
'willow_tree',
'wolf',
'woman',
'worm',
]

small_to_big_mapping = {
'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
'people': ['baby', 'boy', 'girl', 'man', 'woman'],
'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


big_to_small_mapping = {'beaver': 'aquatic mammals',
 'dolphin': 'aquatic mammals',
 'otter': 'aquatic mammals',
 'seal': 'aquatic mammals',
 'whale': 'aquatic mammals',
 'aquarium_fish': 'fish',
 'flatfish': 'fish',
 'ray': 'fish',
 'shark': 'fish',
 'trout': 'fish',
 'orchid': 'flowers',
 'poppy': 'flowers',
 'rose': 'flowers',
 'sunflower': 'flowers',
 'tulip': 'flowers',
 'bottle': 'food containers',
 'bowl': 'food containers',
 'can': 'food containers',
 'cup': 'food containers',
 'plate': 'food containers',
 'apple': 'fruit and vegetables',
 'mushroom': 'fruit and vegetables',
 'orange': 'fruit and vegetables',
 'pear': 'fruit and vegetables',
 'sweet_pepper': 'fruit and vegetables',
 'clock': 'household electrical device',
 'computer_keyboard': 'household electrical device',
 'lamp': 'household electrical device',
 'telephone': 'household electrical device',
 'television': 'household electrical device',
 'bed': 'household furniture',
 'chair': 'household furniture',
 'couch': 'household furniture',
 'table': 'household furniture',
 'wardrobe': 'household furniture',
 'bee': 'insects',
 'beetle': 'insects',
 'butterfly': 'insects',
 'caterpillar': 'insects',
 'cockroach': 'insects',
 'bear': 'large carnivores',
 'leopard': 'large carnivores',
 'lion': 'large carnivores',
 'tiger': 'large carnivores',
 'wolf': 'large carnivores',
 'bridge': 'large man-made outdoor things',
 'castle': 'large man-made outdoor things',
 'house': 'large man-made outdoor things',
 'road': 'large man-made outdoor things',
 'skyscraper': 'large man-made outdoor things',
 'cloud': 'large natural outdoor scenes',
 'forest': 'large natural outdoor scenes',
 'mountain': 'large natural outdoor scenes',
 'plain': 'large natural outdoor scenes',
 'sea': 'large natural outdoor scenes',
 'camel': 'large omnivores and herbivores',
 'cattle': 'large omnivores and herbivores',
 'chimpanzee': 'large omnivores and herbivores',
 'elephant': 'large omnivores and herbivores',
 'kangaroo': 'large omnivores and herbivores',
 'fox': 'medium-sized mammals',
 'porcupine': 'medium-sized mammals',
 'possum': 'medium-sized mammals',
 'raccoon': 'medium-sized mammals',
 'skunk': 'medium-sized mammals',
 'crab': 'non-insect invertebrates',
 'lobster': 'non-insect invertebrates',
 'snail': 'non-insect invertebrates',
 'spider': 'non-insect invertebrates',
 'worm': 'non-insect invertebrates',
 'baby': 'people',
 'boy': 'people',
 'girl': 'people',
 'man': 'people',
 'woman': 'people',
 'crocodile': 'reptiles',
 'dinosaur': 'reptiles',
 'lizard': 'reptiles',
 'snake': 'reptiles',
 'turtle': 'reptiles',
 'hamster': 'small mammals',
 'mouse': 'small mammals',
 'rabbit': 'small mammals',
 'shrew': 'small mammals',
 'squirrel': 'small mammals',
 'maple_tree': 'trees',
 'oak_tree': 'trees',
 'palm_tree': 'trees',
 'pine_tree': 'trees',
 'willow_tree': 'trees',
 'bicycle': 'vehicles 1',
 'bus': 'vehicles 1',
 'motorcycle': 'vehicles 1',
 'pickup_truck': 'vehicles 1',
 'train': 'vehicles 1',
 'lawn_mower': 'vehicles 2',
 'rocket': 'vehicles 2',
 'streetcar': 'vehicles 2',
 'tank': 'vehicles 2',
 'tractor': 'vehicles 2'}


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 1')
    
    sheet1.write(0,0,'Actual label')
    sheet1.write(0,1,'Actual label''s category')
    sheet1.write(0,2,'Actual_number (0 index)')
    
    sheet1.write(0,3,'Predicted label')
    sheet1.write(0,4,'Predicted label''s category')
    sheet1.write(0,5,'Predicted_number (0 index)')
  
    i = 1

    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()
    
        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
        actual_label = labels.item()
        prediction_label = pred.item()
        sheet1.write(i,0, coarse_label[actual_label])
        sheet1.write(i,1, big_to_small_mapping[coarse_label[actual_label]])
        sheet1.write(i,2, actual_label)

        sheet1.write(i,3, coarse_label[prediction_label])
        sheet1.write(i,4, big_to_small_mapping[coarse_label[prediction_label]])
        sheet1.write(i,5, prediction_label)
        
        i +=1

    wb.save('predictions.xls') 

    
    val_acc = correct / total
    return val_acc



    

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=8)


print('Finish loading test data')

test_acc = test(test_loader)

print('test_acc: %.3f' % (test_acc))

