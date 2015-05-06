import time
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter  	# for easiness in sorting and finding max and stuff
import datetime
import numpy as np 	# many operations are done in numpy as matrix inverse; for efficiency
from MAB_algorithms import LinUCBAlgorithm
from util_functions import Stats

class BlindedLinUCBAlgorithm(LinUCBAlgorithm):
	def __init__(self, dimension, alpha):
		super(BlindedLinUCBAlgorithm, self).__init__(dimension, alpha)
		
		self.bt = np.random.binomial(1, .5)
		self.btPrevious = np.random.binomial(1, .5)
		self.btNext = np.random.binomial(1, .5)
		self.lastAction = None

	def decide(self, pool_articles, user, time_):

		if (self.btPrevious==0 and slef.bt==1) or self.lastAction is None:
			articlePicked = super(BlindedLinUCBAlgorithm, self).decide(
				pool_articles, user, time_)
		else:
			articlePicked = self.lastAction

		self.lastAction = articlePicked
		return articlePicked

	def updateParameters(self, pickedArticle, userArrived, click, time_):
		if self.bt == 0 and self.btNext ==1:
			super(BlindedLinUCBAlgorithm, self).updateParameters(
				pickedArticle, userArrived, click, time_)

		self.btPrevious = self.bt 
		self.bt = self.btNext
		self.btNext = np.random.binomial(1, .5)

