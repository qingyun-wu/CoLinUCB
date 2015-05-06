import time
import re   # regular expression library
from random import random, choice, randint  #from random strategy
from operator import itemgetter      #for easiness in sorting and finding max and stuff
import datetime
import numpy as np 
from util_functions import Stats
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize
class UserStruct:
	def __init__(self, featureDimension, userID, lambda_):
		self.userID = userID
		self.A = lambda_*np.identity(n = featureDimension)
		self.b = np.zeros(featureDimension)
		self.UserTheta = np.zeros(featureDimension)

		self.pta = 0.0
		self.mean = 0.0
		self.var = 0.0
	
	def updateParameters(self, featureVector, click, W, users):
		self.A  +=np.outer(featureVector, featureVector)
		self.b += featureVector*click
		self.updateUserTheta(W, users)

	def updateUserTheta(self, W, users):
		i = self.userID
		AComplex =  sum([W[i][j]*users[j].A for j in range(W.shape[1])])
		bComplex =  sum([W[i][j]*users[j].b for j in range(W.shape[1])])
		
		self.UserTheta = np.dot(np.linalg.inv(AComplex), bComplex)
		'''
		if int(i) == 1:
			print 'User1_Estimated theta', self.UserTheta
		'''
	def getProb(self, alpha, W, users, featureVector):
		i = self.userID
		TempTheta = sum([W[i][j]*users[j].UserTheta for j in range(W.shape[1])])
		TempA = sum([(W[i][j]**2) * np.linalg.inv(users[j].A) for j in range(W.shape[1])])

		self.mean = np.dot(TempTheta, featureVector)
		self.var = np.sqrt(np.dot(np.dot(featureVector, TempA), featureVector))
		self.pta = self.mean + alpha * self.var
		return self.pta


class CoLinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n):  # n is number of users
		self.users = {} 
		for i in range(n):
			self.users[i] = UserStruct(dimension, i, lambda_ )
		self.W = self.initilizeW()
		self.dimension = dimension

		self.alpha = alpha
		self.lambda_ = lambda_

	def decide(self, pool_articles, userID, time_):
		#n = len(self.users)		
		maxPTA = float('-inf')
		articlePicked =choice(pool_articles)
		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, self.W, self.users,  x.featureVector)
			
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked.featureVector, click, self.W, self.users)

	def initilizeW(self):
		n = len(self.users)
		#----------Sparse-------
		a = np.ones(n-1) 
		b =np.ones(n);
		c = np.ones(n-1)
		k1 = -1
		k2 = 0
		k3 = 1
		A = np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
		G = A
		#----------Sparse-------
		L = csgraph.laplacian(G, normed = False)
		epsilon = 0.2
		I = np.identity(n)
		W = I - epsilon * L  # W1 is a double stostastic matrix
		return W
	''''
	def getLearntParams(self, userID):
		return self.users[userID].UserTheta
	def getarticleCTR(self, userID, articleID):
		return self.users[userID].KArticles[articleID].pta
	'''
