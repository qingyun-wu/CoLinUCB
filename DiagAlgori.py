import time
import re   # regular expression library
from random import random, choice,randint  #from random strategy
from operator import itemgetter      #for easiness in sorting and finding max and stuff
import datetime
import numpy as np 
from util_functions import Stats
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize
class LinUCBUserStruct(object):
	def __init__(self, featureDimension, userID, lambda_):
		self.userID = userID
		self.A = lambda_*np.identity(n = featureDimension)
		self.b = np.zeros(featureDimension)
		self.UserTheta = np.zeros(featureDimension)
		self.lambda_ = lambda_

		self.pta = 0.0
		self.mean = 0.0
		self.var = 0.0

	def updateParameters(self, featureVector, click):
		self.A += np.outer(featureVector, featureVector)
		self.b += featureVector*click
		self.UserTheta = np.dot(np.linalg.inv(self.A), self.b)

	def getProb(self, alpha, users, featureVector):
		self.mean = np.dot(users[self.userID].UserTheta, featureVector)
		self.var = np.sqrt(np.dot(np.dot(featureVector, np.linalg.inv(users[self.userID].A)), featureVector))
		self.pta = self.mean + alpha * self.var
		return self.pta


class CoLinUCBUserStruct(LinUCBUserStruct):
	def __init__(self, featureDimension, userID, lambda_):
		LinUCBUserStruct.__init__(self, featureDimension, userID, lambda_)

	def getProb(self, alpha, users, featureVector, W):
		i = self.userID
		TempTheta = sum([W[i][j]*users[j].UserTheta for j in range(W.shape[1])])

		TempI = self.lambda_ *np.identity(n = self.A.shape[1])
		
		TempA = sum([(W[i][j]**4) * (users[j].A - TempI) for j in range(W.shape[1])]) + TempI
		
		self.mean = np.dot(TempTheta, featureVector)
		#self.var = np.sqrt(np.dot(np.dot(featureVector, np.linalg.inv(users[i].A )), featureVector))
		self.var = np.sqrt(np.dot(np.dot(featureVector,  np.linalg.inv(TempA)), featureVector))
		self.pta = self.mean + alpha * self.var

		return self.pta	

	def updateParameters(self, featureVector, click, users, W):
		i = self.userID
			
		
		#ONLY take diagonal
		TempX = np.outer(featureVector, featureVector)
		self.A += sum([W[m][m]**2 *TempX for m in range(W.shape[0])])

		TempTheta = sum([W[i][j]* users[j].UserTheta for j in range(W.shape[1])]) - W[i][i]*users[i].UserTheta
		Temp = featureVector*click -  np.dot(np.outer(featureVector, featureVector), TempTheta)
		self.b +=W[i][i]* Temp
		
		'''
		TempX = np.outer(featureVector, featureVector)
		self.A += sum([(W[m][i]**2)*TempX for m in range(W.shape[0])])
			
		TempTheta  = np.zeros(users[i].UserTheta.shape[0])
		for m in range(W.shape[0]):
			TempTheta += W[m][i] * (sum([W[m][j]* users[j].UserTheta for j in range(W.shape[1])]) - W[m][i]*users[i].UserTheta)
		self.b += featureVector*click -  np.dot(np.outer(featureVector, featureVector), TempTheta)
		'''

		
		self.UserTheta = np.dot(np.linalg.inv(self.A), self.b)

class LinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, decay = None):  # n is number of users
		self.users = []
		for i in range(n):
			self.users.append(LinUCBUserStruct(dimension, i, lambda_ )) 
		self.dimension = dimension

		self.alpha = alpha

	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, self.users, x.featureVector)

			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked.featureVector, click)
		#print 'user', userID, "UserTheta", self.users[userID].UserTheta
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta



class CoLinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, W):  # n is number of users
		self.users = [] 
		for i in range(n):
			self.users.append(CoLinUCBUserStruct(dimension, i, lambda_ )) 
		self.dimension = dimension
		self.alpha = alpha
		self.W = W

	def decide(self, pool_articles, userID):
		#n = len(self.users)		
		maxPTA = float('-inf')
		articlePicked = None
		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, self.users,  x.featureVector, self.W)
			
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked.featureVector, click, self.users, self.W)
		#print 'user', userID, "UserTheta", self.users[userID].UserTheta
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta



