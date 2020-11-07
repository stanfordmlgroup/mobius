import numpy as np
import torch
from PIL import Image
import random 
from scipy.ndimage import geometric_transform, map_coordinates
from numpy import *
from random import random
#np.random.seed(0)

def getabcd_random(height,width):
    from random import random
   
    zp=[complex(height*random(),width*random()), complex(height*random(),width*random()),complex(height*random(),width*random())]; 
    wa=[complex(height*random(),width*random()), complex(height*random(),width*random()),complex(height*random(),width*random())];
    a = np.linalg.det([[zp[0]*wa[0], wa[0], 1], 
                    [zp[1]*wa[1], wa[1], 1], 
                    [zp[2]*wa[2], wa[2], 1]]);
    b = np.linalg.det([[zp[0]*wa[0], zp[0], wa[0]], 
                    [zp[1]*wa[1], zp[1], wa[1]], 
                    [zp[2]*wa[2], zp[2], wa[2]]]);  

    c = np.linalg.det([[zp[0], wa[0], 1], 
                    [zp[1], wa[1], 1], 
                    [zp[2], wa[2], 1]]);

    d = np.linalg.det([[zp[0]*wa[0], zp[0], 1], 
                    [zp[1]*wa[1], zp[1], 1], 
                    [zp[2]*wa[2], zp[2], 1]]);
    
    return a,b,c,d







class Mobius(object):
    def __init__(self,rand,interpolation,dataset, std, madmissable,M):
        if dataset == 'cifar10' or dataset ==  'cifar100':
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
        self.mode='constant'
        e=[complex(0,0)]*self.h*self.w
        self.z=np.array(e).reshape(self.h,self.w)

        for i in range(0,self.h):
            for j in range(0,self.w):
                self.z[i,j]=complex(i,j)
        self.i=np.array(list(range(0,self.h))*self.w).reshape(self.h,self.w).T
        self.j=np.array(list(range(0,self.h))*self.w).reshape(self.h,self.w)
        self.rand = rand
        self.interpolation = interpolation
        self.std = std
        self.madmissable = madmissable
        self.M = M


    def __call__(self, image):

#         if self.interpolation:
#             image = self.mobius_with_interpolation (image)
#         else:
        
        image = self.mobius_fast_interpolation (image,self.std)
#         image = self.mobius_with_interpolation (image, self.std)
          
        return image
    
    def mobius_with_interpolation(self, image, std):
        height=self.h
        width=self.w
#         print(image)
        if self.rand == True:
            a, b, c, d = self.getabcd_1fix(height,width,std)
        elif self.madmissable == True:
            a, b, c, d = self.getabcd_madmissable(height,width)
        else:
            a, b, c, d = self.getabcd_8fix(height,width)
        

            
        r = geometric_transform(image,self.shift_func,cval=0,output_shape=(height,width,3),mode=self.mode,extra_arguments=(a,b,c,d))
        new_image=r
        new_image=Image.fromarray(new_image)
        return new_image
        
    
    def mobius_fast_interpolation(self, image,std):
        image = np.array(image)
        
        height=self.h
        width=self.w
        if self.rand == True:
            a, b, c, d = self.getabcd_1fix(height,width,std)
        elif self.madmissable == True:
            a, b, c, d = self.getabcd_madmissable(height,width)
        else:
            a, b, c, d = self.getabcd_8fix(height,width)
        r = ones((height, width,3),dtype=uint8)*255*0

        z=self.z
        i=self.i
        j=self.j
                
        w = (a*z+b)/(c*z+d)
        first=real(w)*1
        second=imag(w)*1
        first=first.astype(int)
        second=second.astype(int)
        f1=first>=0
        f2=first<height
        
        f= f1 & f2
        s1=second>=0
        s2=second<width
        s= s1 & s2
        combined = s&f

        r[first[combined],second[combined],:]=image[i[combined],j[combined],:]

        
        u=[True]*height*width
        canvas=np.array(u).reshape(height,width)
        canvas[first[combined],second[combined]]=False
        converted_empty_index = np.where(canvas == True )
        converted_first = converted_empty_index[0]
        converted_second = converted_empty_index[1]

        new = converted_first.astype(complex)
        new.imag = converted_second


        ori = (d*new-b)/(-c*new+a)

        p=np.hstack([ori.real,ori.real,ori.real])
        k=np.hstack([ori.imag,ori.imag,ori.imag])
        zero=np.zeros_like(ori.real)
        one=np.ones_like(ori.real)
        two=np.ones_like(ori.real)*2
        third = np.hstack([zero,one,two])
        number_of_interpolated_point = len(one)
        e = number_of_interpolated_point
        interpolated_value_unfinished = map_coordinates(image, [p, k,third], order=1,mode='constant',cval=0)
        t = interpolated_value_unfinished

        interpolated_value = np.stack([t[0:e],t[e:2*e],t[2*e:]]).T

        r[converted_first,converted_second,:] = interpolated_value

       
        new_image=Image.fromarray(r)
      
        return new_image
    
    def getabcd_8fix(self, height, width):

        raffle = np.random.randint(8)
        #print("raffle:",raffle)

        if raffle == 0:
            #clockwise 90 degree turn
            zp=[complex(1,0.5*width), complex(0.5*height,0.8*width), complex(0.6*height,0.5*width)]; 
            wa=[complex(0.5*height,width-1), complex(0.5*height+0.4*width,0.5*width),complex(0.5*height,0.5*width-0.1*height)];
        elif raffle == 1:
        #clockwise twist
            zp=[complex(1,0.5*width), complex(0.5*height,0.8*width), complex(0.6*height,0.5*width)]; 
            wa=[complex(0.5*height,width-1), complex(0.5*height+0.3*width*math.sin(0.5*pi*0.8),0.5*width+0.3*width*math.cos(0.5*pi*0.8)),complex(0.5*height+0.1*height*math.cos(0.5*pi*0.2),0.5*width-0.1*height*math.sin(0.5*pi*0.2))];

        elif raffle == 2:
            # enlarge/ spread
            zp=[complex(0.3*height,0.5*width), complex(0.5*height,0.7*width), complex(0.7*height,0.5*width)]; 
            wa=[complex(0.2*height,0.5*width), complex(0.5*height,0.8*width),complex(0.8*height,0.5*width)];

        elif raffle == 3:
             # enlarge/ spread
            zp=[complex(0.3*height,0.3*width), complex(0.6*height,0.8*width), complex(0.7*height,0.3*width)]; 
            wa=[complex(0.2*height,0.3*width), complex(0.6*height,0.9*width),complex(0.8*height,0.2*width)];

        elif raffle == 4:
            #counter clockwise twist
            wa=[complex(1,0.5*width), complex(0.5*height,0.8*width), complex(0.6*height,0.5*width)]; 
            zp=[complex(0.5*height,width-1), complex(0.5*height+0.4*width,0.5*width),complex(0.5*height,0.5*width-0.1*height)];

        elif raffle == 5:
            #counter clockwise 90 degree turn
            wa=[complex(1,0.5*width), complex(0.5*height,0.8*width), complex(0.6*height,0.5*width)]; 
            zp=[complex(0.5*height,width-1), complex(0.5*height+0.3*width*math.sin(0.5*pi*0.8),0.5*width+0.3*width*math.cos(0.5*pi*0.8)),complex(0.5*height+0.1*height*math.cos(0.5*pi*0.2),0.5*width-0.1*height*math.sin(0.5*pi*0.2))];

        elif raffle == 6:
            #inverse
            zp=[complex(1,0.5*width), complex(0.5*height,0.9*width), complex(height-1,0.5*width)]; 
            wa=[complex(height-1,0.5*width), complex(0.5*height,0.1*width),complex(1,0.5*width)];

        elif raffle == 7:
            #inverse spread
            zp=[complex(0.1*height,0.5*width), complex(0.5*height,0.8*width), complex(0.9*height,0.5*width)]; 
            wa=[complex(height-1,0.5*width), complex(0.5*height,0.1*width),complex(1,0.5*width)];
#         elif raffle == 8:
#             #super spread
#             zp=[complex(0.3*height,0.3*width), complex(0.6*height,0.7*width), complex(0.7*height,0.3*width)]; 
#             wa=[complex(0.1*height,0.3*width), complex(0.6*height,0.9*width),complex(0.9*height,0.3*width)];
            
            
        a = np.linalg.det([[zp[0]*wa[0], wa[0], 1], 
                        [zp[1]*wa[1], wa[1], 1], 
                        [zp[2]*wa[2], wa[2], 1]]);
        b = np.linalg.det([[zp[0]*wa[0], zp[0], wa[0]], 
                        [zp[1]*wa[1], zp[1], wa[1]], 
                        [zp[2]*wa[2], zp[2], wa[2]]]);  

        c = np.linalg.det([[zp[0], wa[0], 1], 
                        [zp[1], wa[1], 1], 
                        [zp[2], wa[2], 1]]);

        d = np.linalg.det([[zp[0]*wa[0], zp[0], 1], 
                        [zp[1]*wa[1], zp[1], 1], 
                        [zp[2]*wa[2], zp[2], 1]]);

        return a,b,c,d
    

    def shift_func(self, coords,a,b,c,d):
        """ Define the moebius transformation, though backwards """
        #turn the first two coordinates into an imaginary number
        z = coords[0] + 1j*coords[1]
        w = (d*z-b)/(-c*z+a) #the inverse mobius transform
        #take the color along for the ride
        return np.real(w),np.imag(w),coords[2]





    
    def getabcd_1fix(self,height,width, std):
        
        raffle = np.random.randint(6)
        if raffle == 0 :

            a=np.random.normal(0.5,std)
            b=np.random.normal(0.25,std)
            c=np.random.normal(0.25,std)
            d=np.random.normal(0.5,std)
            e=np.random.normal(0.5,std)
            f=np.random.normal(0.75,std)
            zp=[complex(0.25*height,0.5*width), complex(0.5*height,0.75*width), complex(0.75*height,0.5*width)]
            wa=[complex(a*height,b*width), complex(c*height,d*width),complex(e*height,f*width)]
            original_points = np.array([[0.25*height,0.5*width], [0.5*height,0.75*width], [0.75*height,0.5*width]],dtype=int)
            new_points  = np.array([[a*height,b*width], [c*height,d*width],[e*height,f*width]],dtype=int)

        elif raffle == 1:
            a=np.random.normal(0.5,std)
            b=np.random.normal(0.75,std)
            c=np.random.normal(0.25,std)
            d=np.random.normal(0.5,std)
            e=np.random.normal(0.5,std)
            f=np.random.normal(0.25,std)
            zp=[complex(0.25*height,0.5*width), complex(0.5*height,0.25*width), complex(0.75*height,0.5*width)]
            wa=[complex(a*height,b*width), complex(c*height,d*width),complex(e*height,f*width)]
            original_points = np.array([[0.25*height,0.5*width], [0.5*height,0.25*width], [0.75*height,0.5*width]],dtype=int)
            new_points  = np.array([[a*height,b*width], [c*height,d*width],[e*height,f*width]],dtype=int)

        elif raffle == 2:
            a=np.random.normal(0.25,std)
            b=np.random.normal(0.75,std)
            c=np.random.normal(0.5,std)
            d=np.random.normal(0.5,std)
            e=np.random.normal(0.75,std)
            f=np.random.normal(0.25,std)
            zp=[complex(0.25*height,0.75*width), complex(0.5*height,0.5*width), complex(0.75*height,0.25*width)]
            wa=[complex(a*height,b*width), complex(c*height,d*width),complex(e*height,f*width)]
            original_points = np.array([[0.25*height,0.75*width], [0.5*height,0.5*width], [0.75*height,0.25*width]],dtype=int)
            new_points  = np.array([[a*height,b*width], [c*height,d*width],[e*height,f*width]],dtype=int) 



        elif raffle == 3:

            a=np.random.normal(0.25,std)
            b=np.random.normal(0.5,std)
            c=np.random.normal(0.5,std)
            d=np.random.normal(0.75,std)
            e=np.random.normal(0.75,std)
            f=np.random.normal(0.5,std)
            zp=[complex(0.25*height,0.5*width), complex(0.5*height,0.75*width), complex(0.75*height,0.5*width)]
            wa=[complex(a*height,b*width), complex(c*height,d*width),complex(e*height,f*width)]
            original_points = np.array([[0.25*height,0.5*width], [0.5*height,0.75*width], [0.75*height,0.5*width]],dtype=int)
            new_points  = np.array([[a*height,b*width], [c*height,d*width],[e*height,f*width]],dtype=int)
        elif raffle == 4:
            a=np.random.normal(0.75,std)
            b=np.random.normal(0.5,std)
            c=np.random.normal(0.5,std)
            d=np.random.normal(0.25,std)
            e=np.random.normal(0.25,std)
            f=np.random.normal(0.5,std)
            zp=[complex(0.25*height,0.5*width), complex(0.5*height,0.75*width), complex(0.75*height,0.5*width)]
            wa=[complex(a*height,b*width), complex(c*height,d*width),complex(e*height,f*width)]
            original_points = np.array([[0.25*height,0.5*width], [0.5*height,0.75*width], [0.75*height,0.5*width]],dtype=int)
            new_points  = np.array([[a*height,b*width], [c*height,d*width],[e*height,f*width]],dtype=int)            
        elif raffle == 5:
            a=np.random.normal(0.75,std)
            b=np.random.normal(0.5,std)
            c=np.random.normal(0.5,std)
            d=np.random.normal(0.75,std)
            e=np.random.normal(0.25,std)
            f=np.random.normal(0.5,std)
            zp=[complex(0.25*height,0.5*width), complex(0.5*height,0.25*width), complex(0.75*height,0.5*width)]
            wa=[complex(a*height,b*width), complex(c*height,d*width),complex(e*height,f*width)]
            original_points = np.array([[0.25*height,0.5*width], [0.5*height,0.25*width], [0.75*height,0.5*width]],dtype=int)
            new_points  = np.array([[a*height,b*width], [c*height,d*width],[e*height,f*width]],dtype=int)


            
        a = np.linalg.det([[zp[0]*wa[0], wa[0], 1], 
                        [zp[1]*wa[1], wa[1], 1], 
                        [zp[2]*wa[2], wa[2], 1]]);
        b = np.linalg.det([[zp[0]*wa[0], zp[0], wa[0]], 
                        [zp[1]*wa[1], zp[1], wa[1]], 
                        [zp[2]*wa[2], zp[2], wa[2]]]);  

        c = np.linalg.det([[zp[0], wa[0], 1], 
                        [zp[1], wa[1], 1], 
                        [zp[2], wa[2], 1]]);

        d = np.linalg.det([[zp[0]*wa[0], zp[0], 1], 
                        [zp[1]*wa[1], zp[1], 1], 
                        [zp[2]*wa[2], zp[2], 1]]);

        return a,b,c,d
    
    def getabcd_madmissable(self,height,width):
        
        test=False #finding true ones
        while test==False:
            zp=[complex(height*random(),width*random()), complex(height*random(),width*random()),complex(height*random(),width*random())] 
            wa=[complex(height*random(),width*random()), complex(height*random(),width*random()),complex(height*random(),width*random())]

            original_points = np.array([[real(zp[0]),imag(zp[0])],
                                    [real(zp[1]),imag(zp[1])],
                                    [real(zp[2]),imag(zp[2])]],dtype=int)
            new_points = np.array([[real(wa[0]),imag(wa[0])],
                                    [real(wa[1]),imag(wa[1])],
                                    [real(wa[2]),imag(wa[2])]],dtype=int)
            # transformation parameters
            a = linalg.det([[zp[0]*wa[0], wa[0], 1], 
                        [zp[1]*wa[1], wa[1], 1], 
                        [zp[2]*wa[2], wa[2], 1]]);

            b = linalg.det([[zp[0]*wa[0], zp[0], wa[0]], 
                        [zp[1]*wa[1], zp[1], wa[1]], 
                        [zp[2]*wa[2], zp[2], wa[2]]]);         


            c = linalg.det([[zp[0], wa[0], 1], 
                        [zp[1], wa[1], 1], 
                        [zp[2], wa[2], 1]]);

            d = linalg.det([[zp[0]*wa[0], zp[0], 1], 
                        [zp[1]*wa[1], zp[1], 1], 
                        [zp[2]*wa[2], zp[2], 1]]);
            test=self.M_admissable(a,b,c,d)

        return a,b,c,d
    def M_admissable(self,a,b,c,d):
        M=self.M
        size = self.h
#         size = 32
        v1 = np.absolute(a) ** 2 / np.absolute(a*d - b*c)
        if not (v1 < M and v1 > 1/M):
            return False

        v2 = np.absolute(a-size*c) ** 2 / (np.absolute(a*d -b*c))
        if not (v2 < M and v2 > 1/M):
            return False

        v3 = np.absolute(complex(a,-size*c)) ** 2 / np.absolute(a*d-b*c)
        if not (v3 < M and v3 > 1/M):
            return False

        v4 = np.absolute(complex(a-size*c,-size*c)) ** 2 / np.absolute(a*d-b*c)
        if not (v4 < M and v4 > 1/M):
            return False

        v5 = np.absolute(complex(a-size/2*c,-size/2*c)) ** 2 / (np.absolute(a*d-b*c))
        if not (v5 < M and v5 > 1/M):
            return False

        v6 = np.absolute(complex(size/2*d-b,size/2*d)/complex(a-size/2*c,-size/2*c)-complex(size/2,size/2))
        if not( v6 < size/4):
            return False


        return  True

