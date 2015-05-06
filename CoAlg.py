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

	def updateParameters(self, articlePicked, click):
		featureVector = articlePicked.featureVector
		self.A += np.outer(featureVector, featureVector)
		self.b += featureVector*click
		self.UserTheta = np.dot(np.linalg.inv(self.A), self.b)

	def getProb(self, alpha, users, article):
		featureVector = article.featureVector
		self.mean = np.dot(users[self.userID].UserTheta, featureVector)
		self.var = np.sqrt(np.dot(np.dot(featureVector, np.linalg.inv(users[self.userID].A)), featureVector))
		self.pta = self.mean + alpha * self.var
		return self.pta


class CoLinUCBUserStruct(LinUCBUserStruct):
	def __init__(self, featureDimension, userID, lambda_):
		LinUCBUserStruct.__init__(self, featureDimension, userID, lambda_)
		self.APFeatureVector =np.zeros(featureDimension)   # ArticlePicked
		self.APreward = 0.0
		
	def updateReward(self, articlePicked, click):
		self.APFeatureVector = articlePicked.featureVector
		self.APreward = click


	def updateParameters(self, articlePicked, click, users, W):
		U_id = self.userID
		'''
		self.APFeatureVector = articlePicked.featureVector
		self.APreward = click
		'''

		self.A += sum([(W[m][U_id]**2) * np.outer(users[m].APFeatureVector, users[m].APFeatureVector) for m in range(W.shape[0])])

		Tempb = np.zeros(articlePicked.featureVector.shape[0])
		for m in range(W.shape[0]):
			TempTheta =(sum([W[m][j]* users[j].UserTheta for j in range(W.shape[1])]) - W[m][U_id]*users[U_id].UserTheta)
			Tempb += W[m][U_id] *(users[m].APFeatureVector * users[m].APreward - np.dot(np.outer(users[m].APFeatureVector, users[m].APFeatureVector), TempTheta) )

		self.b += Tempb
		self.UserTheta = np.dot(np.linalg.inv(self.A), self.b)

	def getProb(self, alpha, users, article, W):
		U_id = self.userID
		CoTheta = sum([W[U_id][j]*users[j].UserTheta for j in range(W.shape[1])])
			
		#TempI = self.lambda_ *np.identity(n = self.A.shape[1])
		#TempA = sum([(W[U_id][j]**4) * (users[j].A - TempI) for j in range(W.shape[1])]) + TempI
		self.mean = np.dot(CoTheta, article.featureVector)
		self.var = np.sqrt(np.dot(np.dot(article.featureVector,  np.linalg.inv(self.A)), article.featureVector))
		self.pta = self.mean + alpha * self.var
		
		return self.pta	



class LinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, decay = None):  # n is number of users
		self.users = []
		for i in range(n):
			self.users.append(LinUCBUserStruct(dimension, i, lambda_ )) 
		self.dimension = dimension

		self.alpha = alpha

	def decide(self, pool_articles, userID):
		#n = len(self.users)		
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, self.users, x)

			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		#print 'CoLinUCB', 'articlePicked', self.users[userID].articlePicked.id
		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click)
		#print 'user', userID, "UserTheta", self.users[userID].UserTheta
	def updateReward(self, articlePicked, click, userID):
		a = 1
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
			x_pta = self.users[userID].getProb(self.alpha, self.users, x, self.W)
			
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click, self.users, self.W)
		#print 'user', userID, "UserTheta", self.users[userID].UserTheta
	def updateReward(self, articlePicked, click, userID):
		self.users[userID].updateReward(articlePicked, click)
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta



