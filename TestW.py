import math
import numpy as np 

#from CoLinUCB import *
#from LinUCB import *
from Algori import *

#from DiagAlgori import *
#from CoAlg import *
#from exp3_MAB import *
#from exp3_MultiUser import *
from matplotlib.pylab import *
from random import random, sample, randint
#import random
from scipy.stats import lognorm
from util_functions import *
from Articles2 import *
from Users import *
import json
from conf import sim_files_folder, result_folder, save_address
import os
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize

n=10
a = np.ones(n-1) 
b =np.ones(n);
c = np.ones(n-1)
k1 = -1
k2 = 0
k3 = 1
A = np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
G = A 
print G
#----------Sparse-------
#G = np.arange(n)*np.arange(n)[:, np.newaxis]   #generate a graph
L = csgraph.laplacian(G, normed = False)
epsilon = 0.001
I = np.identity(n)
W1 = I - epsilon * L  # W1 is a double stostastic matrix

Min = min(W1.diagonal())
W2 = W1 - I * Min*1.1
W = normalize(W2, axis = 1, norm = 'l1')
print W1
    