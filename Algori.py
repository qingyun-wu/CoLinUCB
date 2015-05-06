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

		self.LambdaIdentity = lambda_*np.identity(n = featureDimension)
		self.A = np.zeros(shape = (featureDimension, featureDimension))

		self.CoA = lambda_*np.identity(n = featureDimension)
		self.Cob = np.zeros(featureDimension)

	def updateParameters(self, articlePicked, click, users, W):
		featureVector = articlePicked.featureVector

		self.A += np.outer(featureVector, featureVector)
		self.b += featureVector*click

		U_id = self.userID
		self.CoA = self.LambdaIdentity + sum([ (W[m][U_id]**2)* users[m].A for m in range(W.shape[0])])

		Tempb = np.zeros(self.b.shape[0])
		for m in range(W.shape[0]):
			NeighborTheta =(sum([W[m][j]* users[j].UserTheta for j in range(W.shape[1])]) - W[m][U_id]*users[U_id].UserTheta)
			Tempb += W[m][U_id] *(users[m].b- np.dot(users[m].A, NeighborTheta) )

		self.Cob = Tempb
		self.UserTheta = np.dot(np.linalg.inv(self.CoA), self.Cob)

	def getProb(self, alpha, users, article, W):
		U_id = self.userID
		CoTheta = sum([W[U_id][j]*users[j].UserTheta for j in range(W.shape[1])])
		CCA = sum([W[U_id][j]* users[j].CoA  for j in range(W.shape[1])])
			
		self.mean = np.dot(CoTheta, article.featureVector)
		self.var = np.sqrt(np.dot(np.dot(article.featureVector, np.linalg.inv(CCA)), article.featureVector))
		#self.pta = self.mean + alpha * self.var
		self.pta = self.mean
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
			x_pta = self.users[userID].getProb(self.alpha, self.users,  x, self.W)
			
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click, self.users, self.W)
		#print 'user', userID, "UserTheta", self.users[userID].UserTheta
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta
	def getCoThetaFromCoLinUCB(self, userID):
		return sum([self.W[userID][j]*self.users[j].UserTheta for j in range(self.W.shape[1])])



