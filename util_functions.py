from collections import Counter
from math import log
import numpy as np 
from random import *
from custom_errors import FileExists 

class Stats():
	def __init__(self):
		self.accesses = 0.0    #times the aritcle was chosen to be presented as the best article
		self.clicks  = 0.0   # of times the article was actually clicked by the users
		self.CTR = 0.0       # CTR as calulated by the updateCTR fucntion

	def updateCTR(self):
		try:
			self.CTR = self.clicks/self.accesses
		except ZeroDivisionError:
			self.CTR = 0.0
		return self.CTR
	def addrecord(self, click):
		self.clicks += click
		self.accesses += 1
		self.updateCTR()

class batchAlgorithmStats():
	def __init__(self):
		self.stats = Stats()
		self.clickArray = []
		self.accessArray = []
		self.CTRArray = []
		self.time_ = []
		self.poolMSE = []
		self.articlesCTR = {}  ####### What is this for?
		self.articlesPicked_temp = []
		self.entropy = []
		self.regret = []
	def addRecord(self, iter_, poolMSE, poolArticles):
		self.clickArray.append(self.stats.clicks)
		self.accessArray.append(self.stats.accesses)
		self.CTRArray.append(self.stats.CTR)
		self.time_.append(iter_)
		self.poolMSE.append(poolMSE)
		for x in poolArticles:
			if x in self.articlesCTR:
				self.articlesCTR[x].append(poolArticles[x])
			else:
				self.articlesCTR[x] = [poolArticles[x]]
		self.entropy.append(calculateEntropy(self.articlesPicked_temp))
		self.articlesPicked_temp = []

	def iterationRecord(self, click, articlePicked):
		self.stats.addrecord(click)
		self.articlesPicked_temp.append(articlePicked)
	def plotArticle(self, article_id):
		plot(self.time_, self.articlesCTR[article_id])
		xlabel('Iterations')
		ylabel('CTR')
		title('')

def calculateEntropy(array):
	counts = 1.0 * np.array(map(lambda x: x[1], Counter(array).items()))
	counts = counts / sum(counts)
	entropy = sum([-x*log(x) for x in counts])
	return entropy

def gaussianFeature(dimension, argv):
	mean = argv['mean'] if 'mean' in argv else 0
	std = argv['std'] if 'std' in argv else 1

	mean_vector = np.ones(dimension)*mean
	stdev = np.identity(dimension)*std
	vector = np.random.multivariate_normal(np.zeros(dimension), stdev)

	l2_norm = np.linalg.norm(vector, ord = 2)
	if 'l2_limit' in argv and l2_norm > argv['l2_limit']:
		"This makes it uniform over the circular range"
		vector = (vector / l2_norm)
		vector = vector * (random())
		vector = vector * argv['l2_limit']

	if mean is not 0:
		vector = vector + mean_vector

	vectorNormalized = []
	for i in range(len(vector)):
		vectorNormalized.append(vector[i]/sum(vector))
	return vectorNormalized
	#return vector

def featureUniform(dimension, argv = None):
	vector = np.array([random() for _ in range(dimension)])
	vectorNormalized = []

	l2_norm = np.linalg.norm(vector, ord =2)
	'''
	if argv and 'l2_limit' in argv and l2_norm > argv['l2_limit']:
		while np.linalg.norm(vector, ord = 2) > argv['l2_limit']:
			vector = np.array([random() for _ in range(dimension)])
	'''
	vector = vector/l2_norm
	return vector

def getBatchStats(arr):
	return np.concatenate((np.array([arr[0]]), np.diff(arr)))

def checkFileExists(filename):
	try:
		with open(filename, 'r'):
			return 1
	except IOError:
		return 0 

def fileOverWriteWarning(filename, force):
	if checkFileExists(filename):
		if force == True:
			print "Warning : fileOverWriteWarning %s"%(filename)
		else:
			raise FileExists(filename)
