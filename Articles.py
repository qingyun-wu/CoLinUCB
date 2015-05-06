import cPickle
import numpy as np 
from util_functions import calculateEntropy, featureUniform, gaussianFeature, fileOverWriteWarning
from random import sample

class Article():	
	def __init__(self, id, startTime, endTime, FV=None):
		self.id = id
		self.startTime = startTime
		self.endTime = endTime
		self.initialTheta = None
		self.theta = None
		self.featureVector = FV
		self.absDiff = {}
		self.time_ = {}
		self.testVars = {}

	def inPool(self, curr_time):
		return curr_time <= self.endTime and curr_time >= self.startTime

	def addRecord(self, time_, absDiff, alg_name):
		if alg_name in self.time_:
			self.time_[alg_name].append(time_)
		else:
			self.time_[alg_name] = [time_]

		if alg_name in self.absDiff:
			self.absDiff[alg_name].append(absDiff)
		else:
			self.absDiff[alg_name] = [absDiff]

	def plotAbsDiff(self):
		figure()
		for k in self.time_.keys():
			plot(self.time_[k], self.absDiff[k])
		legend(self.time_.keys(), loc = 2)
		xlabel("Iterations")
		ylable("Abs Difference between learnt and True parameters")
		title("Observing Learnt parameters Difference")


class ArticleManager():
	def __init__(self, iterations, dimension, thetaFunc, argv, n_articles, poolArticles, influx = 5):
		self.iterations = iterations
		self.dimension = dimension
		self.signature = ""
		self.n_articles = n_articles
		self.poolArticles = poolArticles
		self.thetaFunc = thetaFunc
		self.argv = argv
		self.signature = "A-"+str(self.n_articles)+"+PA"+ str(self.poolArticles)+"+TF-"+self.thetaFunc.__name__
		self.influx = influx
	def saveArticles(self, Articles, filename, force = False):
		fileOverWriteWarning(filename, force)
		with open(filename, 'w') as f:
			cPickle.dump(Articles, f)

	def loadArticles(self, filename):
		with open(filename, 'r') as f:
			return cPickle.load(f)

	def simulateArticlePool(self):
		def getEndTimes():
			pool = range(self.poolArticles)
			endTimes = [0 for i in startTimes]
			last = self.poolArticles
			for i in range(1, intervals):
				chosen = sample(pool, self.influx)
				for c in chosen:
					endTimes[c] = intervalLength * i 
				pool = [x for x in pool if x not in chosen]
				pool +=[x for x in range(last, last + self.influx)]
				last += self.influx    #NO Influx
			for p in pool:
				endTimes[p] = self.iterations
			return endTimes

		articles = []
		articles_id = range(self.n_articles)

		if self.poolArticles and self.poolArticles < self.n_articles:
			remainingArticles = self.n_articles - self.poolArticles
			intervals = remainingArticles / self.influx + 1
			intervalLength = self.iterations / intervals
			startTimes = [0 for x in range(self.poolArticles)] + [
				(1+ int(i/self.influx))*intervalLength for i in range(remainingArticles)]
			endTimes = getEndTimes()

		else:
			startTimes = [0 for x in range(self.n_articles)]
			endTimes = [self.iterations for x in range(self.n_articles)]

		for key, st, ed in zip(articles_id, startTimes, endTimes):
			articles.append(Article(key, st, ed, featureUniform(self.dimension, {})))
			articles[-1].theta = self.thetaFunc(self.dimension, argv = self.argv)
		return articles

