# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:20:35 2015

@author: Qingyun
"""

from scipy.sparse import csgraph
import numpy as np
from sklearn.preprocessing import normalize


n= 10
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

a = np.ones(n-1) 
b =np.ones(n);
c = np.ones(n-1)
A = tridiag(a, b, c)
#a = np.ones(2)
print A

G = A
#G = np.arange(5)* np.arange(5)[:, np.newaxis]  # generate a graph
print G
L = csgraph.laplacian(G, normed = False)
epsilon = 0.2
I = np.identity(n)
W1 = I - epsilon*L     # W1 is a double stostastic matrix, but the diagonal is negative
Min = min(W1.diagonal())
W2 = W1 - I * Min
W = normalize(W2, axis = 1, norm = 'l1')

print W1


'''

def tridiagG(G1, G2, G3, G4,G5, k1=-2, k2=1, k3=0, k4 = 1, k5 = 2):
    return np.diag(G5, k1) + np.diag(G3, k2) + np.diag(G1, k3) +np.diag(G2, k4) + np.diag(G4, k5)

n=5
G1 = np.ones(n)
G2 = np.ones(n-1)
G3 = np.ones(n-1)
G4= np.ones(n-2)
G5= np.ones(n-2)
G = tridiagG(G1, G2, G3, G4,G5)
print G
L = csgraph.laplacian(G, normed = False)
epsilon = 0.2
I = np.identity(5)
W1 = I - epsilon*L     # W1 is a double stostastic matrix, but the diagonal is negative
Min = min(W1.diagonal())
W2 = W1 - I * Min
W = normalize(W2, axis = 1, norm = 'l1')
print W

print A
C = np.ones(3)
SS= A * C
CCC = np.dot(A,C)
print SS
print CCC
'''