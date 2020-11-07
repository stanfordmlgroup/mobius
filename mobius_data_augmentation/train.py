import os
import argparse
import copy
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from numpy.random import randint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.models as models
from torchvision.utils import make_grid
from torchvision import datasets, transforms

from mobius_transformation import Mobius
from transformation import super_transformation
from util.misc import CSVLogger
from util.cutout import Cutout
from model.resnet import ResNet18, ResNet50
from model.wide_resnet import WideResNet


model_options = ['resnet18', 'wideresnet','resnet50','wideresnet2816','densenet121','resnext50']
dataset_options = ['cifar10', 'cifar100', 'tiny','imagenet','stl10','oxfordpet','pet']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar100',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='wideresnet',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--name', type=str, default='debug',
                    help='name of this run')
parser.add_argument('--note', type=str, default='nothing',
                    help='note of this run')
# parser.add_argument('--mobius', type=str, default='true',
#                     help='apply mobius')
parser.add_argument('--rand', action='store_true', default=False,
                    help='apply random mobius')
parser.add_argument('--interpolation', action='store_true', default=True,
                    help='apply interpolation on mobius')
parser.add_argument('--data_augmentation', type=str, default='noaug',
                    help='layer, onlymobius, halfmobius, regular(no mobius), mix, noaug')
parser.add_argument('--pretrain_augmentation', type=str, default='regular',
                    help='layer, onlymobius, halfmobius, regular(no mobius), mix, noaug')
parser.add_argument('--pretrain_number', type=int, default=0,
                    help='number of pretrained epoch')
parser.add_argument('--subset', action='store_true', default=False,
                    help='use less data to train')
parser.add_argument('--subset_size', type=int, default=3,
                    help='use how many data to train in subset')

parser.add_argument('--lamb', default=0.01, type=float,
                    help='lambda for unsupervised loss')
parser.add_argument('--unsup_loss',  type=str, default='kl_div',
                    help='MSE or kl_div')
parser.add_argument('--when_unlabel', type=int, default=199,
                    help='when to start using unlabeled data')
parser.add_argument('--mask', action='store_true', default=False,
                    help='if to use mask')
parser.add_argument('--mask_length', type=int, default=10,
                    help='length of mask(before mobius)')
parser.add_argument('--std', type=float, default=0.01,
                    help='std for restricted random mobius')
parser.add_argument('--madmissable', action='store_true', default=False,
                    help='if to use madmissable')
parser.add_argument('--M', type=float, default=3,
                    help='M for madmissable')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.dataset + '_' + args.model + '_' + args.name
print(args)

# Image Preprocessing
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    train_transform = super_transformation(args.rand,args.interpolation, args.data_augmentation, normalize ,'svhn' , args.n_holes, args.length, args.cutout , args.mask, mask_length=args.mask_length, std =  args.std, madmissable=args.madmissable, M=args.M)
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

elif args.dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = super_transformation(args.rand,args.interpolation, args.data_augmentation, normalize ,'cifar10' , args.n_holes, args.length, ifcutout=args.cutout,ifmask =args.mask, mask_length=args.mask_length, std = args.std,madmissable=args.madmissable, M=args.M)
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

elif args.dataset == 'cifar100':
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = super_transformation(args.rand,args.interpolation, args.data_augmentation, normalize ,'cifar100' , args.n_holes, args.length, ifcutout=args.cutout,ifmask =args.mask,mask_length=args.mask_length, std =  args.std,madmissable=args.madmissable, M=args.M)
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

elif args.dataset == 'imagenet':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = super_transformation(args.rand,args.interpolation, args.data_augmentation, normalize, 'imagenet', args.n_holes, args.length, ifcutout=args.cutout,ifmask =args.mask, mask_length=args.mask_length, std = args.std,madmissable=args.madmissable, M=args.M)
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

elif args.dataset == 'tiny':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = super_transformation(args.rand,args.interpolation, args.data_augmentation, normalize, 'tiny', args.n_holes, args.length, args.cutout, ifmask =args.mask, mask_length=args.mask_length, std = args.std,madmissable=args.madmissable, M=args.M)
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])   

elif args.dataset == 'stl10':
    normalize = transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0)
    train_transform = super_transformation(args.rand,args.interpolation, args.data_augmentation, normalize ,'stl10' , args.n_holes, args.length, ifcutout=args.cutout,ifmask =args.mask,mask_length=args.mask_length, std =  args.std,madmissable=args.madmissable, M=args.M)
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

elif args.dataset == 'pet':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    train_transform = super_transformation(args.rand,args.interpolation, args.data_augmentation, normalize ,'pet' , args.n_holes, args.length, ifcutout=args.cutout,ifmask =args.mask,mask_length=args.mask_length, std =  args.std,madmissable=args.madmissable, M=args.M)
    test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize])   
    
# for unlabeled data points
train_transform2 = transforms.Compose([])
train_transform2.transforms.append(transforms.ToTensor())
train_transform2.transforms.append(normalize)

if args.dataset == 'cifar10':
    num_classes = 10

    train_dataset1 = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    
    train_dataset1 = datasets.CIFAR100(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)

elif args.dataset == 'tiny':
    num_classes = 200
    traindir = 'tiny-imagenet-200/train'
    valdir = 'tiny-imagenet-200/val/images'

    train_dataset1 = datasets.ImageFolder(traindir,
                                  transform=train_transform)
    
    test_dataset = datasets.ImageFolder(valdir,
                                     transform=test_transform)
    
elif args.dataset == 'imagenet':
    num_classes = 1000
    traindir = 'imagenet_32x32/dataset/train'
    valdir = 'imagenet_32x32/dataset/val'
    
    train_dataset1 = datasets.ImageFolder(traindir,
                                  transform=train_transform)
    test_dataset = datasets.ImageFolder(valdir,
                                     transform=test_transform)
elif args.dataset == 'stl10':
    num_classes = 10
    
    train_dataset1 = datasets.STL10(root='stl10_new/',
                                     split='train',
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.STL10(root='stl10_new/',
                                    split='test',
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'pet':
    num_classes = 37
    
    traindir = 'oxfordpet/data_breeds/train'
    valdir = 'oxfordpet/data_breeds/test'
    
    train_dataset1 = datasets.ImageFolder(traindir,
                                  transform=train_transform)
    
    test_dataset = datasets.ImageFolder(valdir,
                                     transform=test_transform)


## use less data to train
if args.subset:
    if args.subset_size == -1 or args.subset_size is None:
        # hacky - only for imagenet
        subset_index = []
        samples_per_class = 1300
        all_index = list(range(0,len(train_dataset1)))
        for i in range(num_classes):
            begin_idx = i * samples_per_class
            subset_index.extend(all_index[begin_idx:begin_idx+600])
        print(f'num subset_indices: len({subset_index})')
    else:
        subset_index = list(range(0,args.subset_size))
    train_dataset1 = torch.utils.data.Subset(train_dataset1,  subset_index)

# Data Loader (Input Pipeline)
train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=8)
                                           #sampler=dist_sampler)
    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=4)

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'resnet50':
    from resnet import resnet50
    cnn = resnet50(num_classes)
    print(cnn)
elif args.model == 'resnext50':
    from resnext import resnext50
    cnn = resnext50(num_classes)
    print(cnn)
elif args.model == 'densenet121':
    from densenet import densenet121
    cnn = densenet121(num_classes)
    print(cnn)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)
elif args.model == 'wideresnet101':
    cnn = torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet101_2', pretrained=False, num_classes=num_classes)

cnn = cnn.cuda()
cnn = torch.nn.DataParallel(cnn).cuda()
criterion = nn.CrossEntropyLoss().cuda()

cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                            momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = CosineAnnealingLR(cnn_optimizer, args.epochs)

filename = 'logs/' + args.dataset+'/'+test_id + '.csv'
os.makedirs('logs/'+ args.dataset+'/', exist_ok=True)
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader):
    cnn.eval() 
    correct = 0.
    total = 0.

    frames = []
    progress_bar_test = tqdm(loader)
    for i, (images, labels) in enumerate(progress_bar_test):
        images = images.cuda()
        labels = labels.cuda()
    
        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
        correctness = pred.data == labels.data
        batch_size = len(labels)
        img_paths = loader.dataset.imgs[i:i+batch_size]
        df = pd.DataFrame(img_paths, columns=['path', 'label'])
        df['pred'] = pred.cpu().detach().numpy()
        df['correctness'] = correctness.cpu().detach().numpy()
        frames.append(df)
    df_merged = pd.concat(frames)

    val_acc = correct / total
    cnn.train()
    return val_acc, df_merged

best_acc = 0
best_epoch = -1
best_model = None
for epoch in range(args.epochs):
    np.random.seed(epoch)
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    accuracy = 0
    progress_bar1 = tqdm(train_loader1)

    for i, (images, labels) in enumerate(progress_bar1):
        progress_bar1.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()
        cnn.zero_grad()
        pred = cnn(images)
        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()

        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar1.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    scheduler.step(epoch)

    test_acc, df_preds = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))
    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)
    
    if test_acc > best_acc:
        
        best_acc = test_acc
        best_epoch = epoch
        if test_acc > 0.1:
            best_model = copy.deepcopy(cnn)

        # keep csv of predictions
        best_preds = df_preds
        preds_dir = 'preds/'+test_id
        os.makedirs(preds_dir, exist_ok=True)
        df_preds.to_csv(f'{preds_dir}/predictions_test_epoch{epoch}_acc{best_acc}.csv')
            
print ("Best test acc:", str(best_acc) )
print("Best Epoch:", str(best_epoch))            
newdir = 'new_checkpoints/'+test_id+ "/"
os.makedirs(newdir, exist_ok=True)
torch.save(cnn.state_dict(), newdir + str(args.epochs)+'epoch' + '.pt')
torch.save(best_model.state_dict(), newdir + 'best_model_at_epoch_'+ str(best_epoch) +'.pt')

csv_logger.close()
best_preds.to_csv(f'{preds_dir}/best_predictions_test_epoch{epoch}_acc{best_acc}.csv', index=False)
