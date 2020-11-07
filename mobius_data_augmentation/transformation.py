import numpy as np
import torch
from PIL import Image
from random import random
from scipy.ndimage import geometric_transform
from numpy import *
from torchvision import transforms
from mobius_transformation import Mobius
from mobius_mask import Mobius_mask
from util.cutout import Cutout
#np.random.seed(0)



class super_transformation(object):
    def __init__(self,rand,interpolation,augmentation_type, normalization, dataset, n_holes, length, ifcutout, ifmask,mask_length,std,madmissable,M):
        
        if dataset == 'cifar10' or  dataset == 'cifar100' or dataset == 'svhn':
            self.h = 32
            self.w = 32            
        elif dataset == 'imagenet':                
            self.h = 32
            self.w = 32
        elif dataset == 'tiny':                
            self.h = 64
            self.w = 64
        elif dataset == 'stl10':                
            self.h = 96
            self.w = 96
        elif dataset == 'pet':                
            self.h = 224
            self.w = 224
        self.augmentation_type = augmentation_type
        self.mobius = Mobius(rand,interpolation,dataset,std, madmissable,M)
        self.mobius_mask = Mobius_mask(dataset,mask_length=mask_length)
        self.flip = transforms.RandomHorizontalFlip()
        self.totensor = transforms.ToTensor()
        self.cutout = Cutout(n_holes=n_holes, length=length)
        self.normalize = normalization
        self.resize = transforms.Resize((self.h,self.w))
        if dataset == 'cifar10' or dataset =='cifar100' or dataset == 'svhn':
            self.crop = transforms.RandomCrop(32, padding=4)
        elif dataset == 'imagenet':                
#             self.crop = transforms.RandomResizedCrop((224,224))
            self.crop = transforms.RandomCrop(32, padding=4)
        elif dataset == 'tiny' :
            self.crop = transforms.RandomCrop(64, padding=4) 
        elif dataset == 'stl10' :
            self.crop = transforms.RandomCrop(96, padding=12)   
        elif dataset == 'pet' :
            self.crop = transforms.CenterCrop(224)
            self.resize = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224)])
            
        self.ifcutout = ifcutout
        self.ifmask = ifmask
    def __call__(self, image):
        if self.augmentation_type == 'noaug':
            # noaug
            image = self.crop(image)
            image = self.totensor(image)
            image = self.normalize(image)
            
        elif self.augmentation_type == 'onlymobius':
            # 100% mobius
            image = self.resize(image)
            image = self.mobius(image)
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type == 'halfmobius':
            # 50% mobius 50% regular
            raffle = np.random.randint(2)
            if raffle == 0: 
                image = self.resize(image)
                image = self.mobius(image)

            else:
                image = self.resize(image)
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type == 'regular':
            image = self.crop(image)
            image = self.flip(image)
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type == 'mix0':
            # 33% mobius
            raffle = random.randint(3)
            if raffle == 0:     
                image = self.resize(image)
                image = self.mobius(image)
            else:   
                image = self.crop(image)
                image = self.flip(image)  
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image) 

        elif self.augmentation_type == 'mix2':
            # 50% mobius
            raffle = random.randint(2)
            if raffle == 0:    
                image = self.resize(image)
                image = self.mobius(image)
            elif raffle == 1:   
                image = self.crop(image)
                image = self.flip(image) 
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type == 'mix3':
            # 20% mobius
            raffle = random.randint(10)
            if raffle < 2:     
                image = self.resize(image)
                image = self.mobius(image)
            else:   
                image = self.crop(image)
                image = self.flip(image)  
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type == 'layer20':
            # 20% mobius
            raffle = random.randint(10)
            if raffle < 2:                     
                image = self.resize(image)
                image = self.mobius(image)
                image = self.crop(image)
                image = self.flip(image) 
            else:   
                image = self.crop(image)
                image = self.flip(image)  
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)  

        elif self.augmentation_type == 'mix4':
            # 10% mobius
            raffle = random.randint(10)
            if raffle == 0:     
                image = self.resize(image)
                image = self.mobius(image)
            else:   
                image = self.crop(image)
                image = self.flip(image)  
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)  

        elif self.augmentation_type == 'mix5':
            # 30% mobius
            raffle = random.randint(10)
            if raffle < 3:     
                image = self.resize(image)
                image = self.mobius(image)
            else:   
                image = self.crop(image)
                image = self.flip(image)  
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image) 

        elif self.augmentation_type == 'mix6':
            # 40% mobius
            raffle = random.randint(10)
            if raffle < 4:     
                image = self.resize(image)
                image = self.mobius(image)
            else:   
                image = self.crop(image)
                image = self.flip(image)  
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)  

        elif self.augmentation_type == 'mix7':
            # 5% mobius
            raffle = random.randint(20)
            if raffle == 0:     
                image = self.resize(image)
                image = self.mobius(image)
            else:   
                image = self.crop(image)
                image = self.flip(image)  
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)  

        elif self.augmentation_type ==  'noaug_mix1':
            raffle = np.random.randint(6)
            if raffle <= 1: 
                image = self.resize(image)
                image = self.mobius(image)
                
            elif raffle <= 4:   
                image = self.crop(image)
                image = self.flip(image)
            else:
                image = self.resize(image)
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type == 'mix' or self.augmentation_type == 'noaug_mix2':
            raffle = np.random.randint(3)
            if raffle == 0: 
                image = self.resize(image)
                image = self.mobius(image)
                
            elif raffle == 1:   
                image = self.crop(image)
                image = self.flip(image)
            else:
                image = self.resize(image)
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type ==  'noaug_mix3':
            raffle = np.random.randint(6)
            if raffle <= 1: 
                #mobius
                image = self.resize(image)
                image = self.mobius(image)
                
            elif raffle == 3:   
                #regular
                image = self.crop(image)
                image = self.flip(image)
            else:
                #noaug
                image = self.resize(image)
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type ==  'noaug_mix4':
            raffle = np.random.randint(5)
            if raffle == 0: 
                #mobius
                image = self.resize(image)
                image = self.mobius(image)                
            elif raffle <= 3:   
                #regular
                image = self.crop(image)
                image = self.flip(image)
            else:
                #noaug
                image = self.resize(image)
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)
            
        elif self.augmentation_type ==  'noaug_mix5':
            raffle = np.random.randint(5)
            if raffle == 0: 
                #mobius
                image = self.resize(image)
                image = self.mobius(image)                
            elif raffle <= 2:   
                #regular
                image = self.crop(image)
                image = self.flip(image)
            else:
                #noaug
                image = self.resize(image)
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type ==  'noaug_mix6':
            raffle = np.random.randint(5)
            if raffle == 0: 
                #mobius
                image = self.resize(image)
                image = self.mobius(image)                
            elif raffle == 1 :   
                #regular
                image = self.crop(image)
                image = self.flip(image)
            else:
                #noaug
                image = self.resize(image)
            image = self.add_mask(image,self.ifmask)
            image = self.totensor(image)
            image = self.normalize(image) 
                        
        elif self.augmentation_type ==  'mask':
            image = self.mobius_mask(image)           
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type ==  'two_mask':
            image = self.mobius_mask(image)#1
            image = self.mobius_mask(image)#2
            image = self.totensor(image)
            image = self.normalize(image)

        elif self.augmentation_type ==  'three_mask':
            image = self.mobius_mask(image)#1
            image = self.mobius_mask(image)#2
            image = self.mobius_mask(image)#3
            image = self.totensor(image)
            image = self.normalize(image)        
        else:
            print("no aug type")
        
        if self.ifcutout:
            image = self.cutout(image)
        
        return image
    
    def add_mask(self, image, ifmask):
        if ifmask == False:
            return image
        else:
            image == self.mobius_mask(image)
            return image
