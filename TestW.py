import math
import numpy as np 

from Algori import *

from matplotlib.pylab import *
from random import random, sample, randint
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

L = csgraph.laplacian(G, normed = False)
epsilon = 0.1
I = np.identity(n)
W = I - epsilon * L  # W is a double stostastic matrix
print W
    