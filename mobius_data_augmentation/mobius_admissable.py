import numpy as np
import random
from collections import defaultdict
from numpy import *

from random import random
counts = defaultdict(int)
height =32
width = 32
M=2
std = 0.05
for i in range(0,1000):
    print(i)
    zp=[complex(height*random(),width*random()), complex(height*random(),width*random()),complex(height*random(),width*random())] 
    wa=[complex(height*random(),width*random()), complex(height*random(),width*random()),complex(height*random(),width*random())]




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



    
    # cond1
    v1 = np.absolute(a) ** 2 / np.absolute(a*d - b*c)
    if not (v1 < M and v1 > 1/M):
        counts['failed'] += 1
        continue

    v2 = np.absolute(a-32*c) ** 2 / (np.absolute(a*d -b*c))
    if not (v1 < M and v1 > 1/M):
        counts['failed'] += 1
        continue

    v3 = np.absolute(complex(a,-32*c)) ** 2 / np.absolute(a*d-b*c)
    if not (v1 < M and v1 > 1/M):
        counts['failed'] += 1
        continue

    v4 = np.absolute(complex(a-32*c,-32*c)) ** 2 / np.absolute(a*d-b*c)
    if not (v1 < M and v1 > 1/M):
        counts['failed'] += 1
        continue

    v5 = np.absolute(complex(a-16*c,-16*c)) ** 2 / (np.absolute(a*d-b*c))
    if not (v1 < M and v1 > 1/M):
        counts['failed'] += 1
        continue
        
    v6 = real(complex(16-b,16*d)/complex(a-16*c,-16*c))
    if not( v6 > 0 and v6 < 32):
        counts['failed'] += 1
        continue
    v7 = imag(complex(16-b,16*d)/complex(a-16*c,-16*c))
    if not( v6 > 0 and v6 < 32):
        counts['failed'] += 1
        continue
        


    counts['passed'] += 1

print(counts)