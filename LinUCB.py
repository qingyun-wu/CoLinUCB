import time
import re   # regular expression library
from random import random, choice,randint  #from random strategy
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

	def updateParameters(self, featureVector, click, users):
		self.A += np.outer(featureVector, featureVector)
		self.b += featureVector*click
		self.updateUserTheta(users)
	
	def updateUserTheta(self, users):
		i = self.userID
		self.UserTheta = np.dot(np.linalg.inv(users[i].A), users[i].b)
		'''
		if int(i) == 1:
			print 'User1_Estimated theta', self.UserTheta
		'''
	def getProb(self, alpha, users, featureVector):
		self.mean = np.dot(users[self.userID].UserTheta, featureVector)
		self.var = np.sqrt(np.dot(np.dot(featureVector, np.linalg.inv(users[self.userID].A)), featureVector))
		self.pta = self.mean + alpha * self.var
		return self.pta
		

class LinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n):  # n is number of users
		self.users = {} 
		for i in range(n):
			self.users[i] = UserStruct(dimension, i, lambda_ )
		self.dimension = dimension

		self.alpha = alpha
		self.lambda_ = lambda_

	def decide(self, pool_articles, userID, time_):
		#n = len(self.users)		
		maxPTA = float('-inf')
		articlePicked =choice(pool_articles)

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, self.users,  x.featureVector)

			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		#print 'CoLinUCB', 'articlePicked', self.users[userID].articlePicked.id
		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked.featureVector, click, self.users)
	'''
	def getLearntParams(self, userID):
		return self.users[userID].UserTheta
	def getarticleCTR(self, userID, articleID):
		return self.users[userID].KArticles[articleID].pta
	'''




