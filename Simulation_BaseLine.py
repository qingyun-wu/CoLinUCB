import math
import numpy as np 
from CoLinUCB import *
from LinUCBimport *
from exp3_MAB import *
from exp3_MultiUser import *
from matplotlib.pylab import *
from random import random, sample, randint
#import random
from scipy.stats import lognorm
from util_functions import *
from Articles2 import *
from Users import *
import json
from conf import sim_files_folder, result_folder, save_address
import os
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize

class simulateOnlineData():
	def __init__(self, dimension, iterations, articles, n_users, userGenerator, userGenerator1, 
		batchSize = 1000,
		noise = lambda : np.random.normal(scale = .001),
		type_ = 'ConstantTheta', environmentVars = None,
		signature = '', poolarticleSize = 10):

		self.simulation_signature = signature
		self.dimension = dimension
		self.type = type_
		self.environmentVars = environmentVars
		self.iterations = iterations
		self.noise = noise
		self.batchSize = batchSize
		self.iter_ = 0
		self.users = {}
		self.poolarticleSize = poolarticleSize

		#Generate all users. Users have two attributes: self.users[id].id, self.users[id].theta
		for i in range(iterations):
			#self.users[i]= userGenerator1.next()
			temp = userGenerator1.next()
			if temp.id not in self.users:
				self.users[temp.id] = temp

		self.W = self.initilizeW()
		self.n_users = n_users     #Number of users

		self.startTime = None
		self.articles = articles   #generate all the articles
		self.articlePool = {}



		"""the temp memory for storing the articles click from expections for each articles"""
		self.articlesPicked = []
		self.alg_perf = {}
		self.reward_vector = {}

		self.userGenerator = userGenerator
		self.initiateEnvironment()

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
		#G = np.arange(n)*np.arange(n)[:, np.newaxis]   #generate a graph
		L = csgraph.laplacian(G, normed = False)
		epsilon = 0.2
		I = np.identity(n)
		W1 = I - epsilon * L  # W1 is a double stostastic matrix
		Min = min(W1.diagonal())
		W2 = W1 - I * Min*1.1
		W = normalize(W2, axis = 1, norm = 'l1')
		return W

	def initiateEnvironment(self):
		env_sign = self.type
		sig_name = [("It",str(self.iterations//1000)+'k')]
		sig_name = '_'.join(['-'.join(tup) for tup in sig_name])
		self.simulation_signature += '_' + env_sign + '_' + sig_name

	def regulateEnvironment(self):
		self.reward_vector = {}
		self.articlePool = [x for x in self.articles if self.iter_ <= x.endTime and self.iter_ >= x.startTime]
	
	def getUser(self):   
		return self.userGenerator.next()

	def getExpectedRewardCoLinUCB(self, UserArrived, pickedArticle):
		tempW = self.W
  
		#print 'W', tempW[UserArrived.id][UserArrived.id], UserArrived.id
		#print 'UserArrived.theta', np.asarray(UserArrived.theta)
		#temp= tempW[UserArrived.id][UserArrived.id] * np.asarray(self.users[UserArrived.id].theta) +sum([tempW[j][UserArrived.id] * np.asarray(self.users[j].theta) for j in range(self.n_users)])
		temp = np.zeros(self.dimension)
		for j in range(self.n_users):
			temp += tempW[j][UserArrived.id] * np.asarray(self.users[j].theta)
		#print 'temp', temp
		#print 'FV', pickedArticle.featureVector
		return np.dot(temp, pickedArticle.featureVector)
	def getExpectedRewardLinUCB(self, userArrived, pickedArticle):   #  For LinUCB
		reward = np.dot(userArrived.theta, pickedArticle.featureVector)
		return reward	

	def GetOptimalArticleCoLinUCB(self, userArrived, articlePool):
		reward ={}

		for x in articlePool:
			 
			reward[x.id] = self.getExpectedRewardCoLinUCB(userArrived, x)
			#print 'articlePool', x.id, x.featureVector, reward[x.id]
			#print 'Optim:', 'userID', userArrived.id, 'userTheta', userArrived.theta,  'Article', x.id, 'ArtFV', x.featureVector, 'Reward', reward[x.id]
		optimalArticle = max([(x, reward[x.id]) for x in articlePool], key = itemgetter(1))[0]
		return optimalArticle

	def GetOptimalArticleLinUCB(self, userArrived, articlePool):
		reward = {}
		for x in articlePool:
			reward[x.id] = self.getExpectedRewardLinUCB(userArrived, x)
		optimalArticle = max([(x, reward[x.id]) for x in articlePool], key = itemgetter(1))[0]
		return optimalArticle 

	def runExp3(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'Exp3'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')

		
		with open(filenameWriteReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccReward')
			f.write('\n')
		with open(filenameWriteOptimalReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccOptimalReward')
			f.write('\n')
		with open(filenameWriteRegret, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'Regret')
			f.write('\n')
			

		tim_ = []
		AccumulatedReward= {}   # dictionary of AccumulatedReward array
		AccumulatedOptimalReward = {}
		AccumulatedRegret = {}
		TempAccReward = {}       # dictionary of currentAccumulatedReward
		TempAccOptimalReward = {} 
		AverageReward =[]
		AverageOptimalReward = []
		AverageRegret = []
		OptimalReward = {}
		click = {}
		#Initialize Reward and OptimalReward
		for i in range(self.n_users):
			AccumulatedReward[i] = []
			AccumulatedOptimalReward[i] = []
			AccumulatedRegret[i] = []
			TempAccReward[i] = 0.0
			TempAccOptimalReward[i] = 0.0
			OptimalReward[i] = 0.0
			click[i] = 0.0 
			

		# About Batch
		BatchAverageReward = []
		BatchAverageOptimalReward = []
		BatchAverageRegret = []

		for self.iter_ in xrange(self.iterations):
			#Generate users by time
			#self.regulateEnvironment()     
			
			#print math.floor(2.3)
			#x = int(math.floor(random()*10))    # generate a userID between 0-10 through random sampling
			#print x

			
			
			for x in self.users:

				#print 'Total Article', len(self.articles)

				#self.articlePool = self.articles[(self.iter_ *len(self.users)*10 + x*10) : (self.iter_ *len(self.users)*10+x*10+10)]
				tempList =[]
				for i in range(self.poolarticleSize):
					tempList.append(self.articles[randint(0,len(self.articles)-1)]) 
				self.articlePool = tempList
				#print 'poolSize', len(self.articlePool)
				
				#print 'user x', x	
				for alg_name, alg in algorithms.items():

					#print 'user x', self.users[x].id, 'true Theta', self.users[x].theta

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleLinUCB(self.users[x], self.articlePool)

					#print 'optimalArticle', self.users[x].id, '  ',  optimalArticle.id
					# The actually chosen article for user x at time t  
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)
					#print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardLinUCB(self.users[x], optimalArticle)
					#print OptimalReward[x]
					TempAccOptimalReward[x] +=OptimalReward[x]
					#print TempAccOptimalReward[x]
					#print 'optimalArticleReward', optimalArticle.id, OptimalReward
					# Compute the actually reward received
					click[x] = self.getExpectedRewardLinUCB( self.users[x], pickedArticle)
					TempAccReward[x] += click[x]

					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccOptimalReward[x] - TempAccReward[x] )				
					if click < 0:
						print click
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))
			
			with open(filenameWriteReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedReward[i][-1] ) for i in range(self.n_users)]))
				f.write(',' + str(AverageReward[-1]))
				f.write('\n')
			with open(filenameWriteOptimalReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedOptimalReward[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageOptimalReward[-1]))
				f.write('\n')
			with open(filenameWriteRegret, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedRegret[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageRegret[-1]))
				f.write('\n')
			#print 'InstantRegret', R[self.iter_]

			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				print BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				print BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				print BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 

		plt.plot(tim_, BatchAverageRegret, label = 'Exp3_Regret')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")
		'''		
		f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = False)
		#ax1.plot(tim_, BatchAverageRegret, label = 'AverageRegret')
		ax1.plot(tim_, BatchAverageReward, label = 'AverageReward')
		ax1.plot(tim_, BatchAverageOptimalReward, label = 'AverageOptimalReward')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Average Regret')
		ax1.set_title('Collaborative LinUCB')
		ax1.legend()
		ax2.plot()
		ax2.set_xlabel('time')
		ax2.set_ylabel('Average Regret Ratio')
		'''


		def plotLines(axes_, xlocs):
			for xloc, color in xlocs:
				# axes = plt.gca()
				for x in xloc:
					xSet = [x for _ in range(31)]
					ymin, ymax = axes_.get_ylim()
					ySet = ymin + (np.array(range(0, 31))*1.0/30) * (ymax - ymin)
					axes_.plot(xSet, ySet, color)

		xlocs = [(list(set(map(lambda x: x.startTime, self.articles))), "black")]
		#plotLines(ax1, xlocs)
		#plotLines(ax2, xlocs)
	def runUCB1(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'UCB1'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')

		
		with open(filenameWriteReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccReward')
			f.write('\n')
		with open(filenameWriteOptimalReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccOptimalReward')
			f.write('\n')
		with open(filenameWriteRegret, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'Regret')
			f.write('\n')
			

		tim_ = []
		AccumulatedReward= {}   # dictionary of AccumulatedReward array
		AccumulatedOptimalReward = {}
		AccumulatedRegret = {}
		TempAccReward = {}       # dictionary of currentAccumulatedReward
		TempAccOptimalReward = {} 
		AverageReward =[]
		AverageOptimalReward = []
		AverageRegret = []
		OptimalReward = {}
		click = {}
		#Initialize Reward and OptimalReward
		for i in range(self.n_users):
			AccumulatedReward[i] = []
			AccumulatedOptimalReward[i] = []
			AccumulatedRegret[i] = []
			TempAccReward[i] = 0.0
			TempAccOptimalReward[i] = 0.0
			OptimalReward[i] = 0.0
			click[i] = 0.0 
			

		# About Batch
		BatchAverageReward = []
		BatchAverageOptimalReward = []
		BatchAverageRegret = []

		for self.iter_ in xrange(self.iterations):
			#Generate users by time
			#self.regulateEnvironment()     
			
			#print math.floor(2.3)
			#x = int(math.floor(random()*10))    # generate a userID between 0-10 through random sampling
			#print x

			
			
			for x in self.users:

				#print 'Total Article', len(self.articles)

				#self.articlePool = self.articles[(self.iter_ *len(self.users)*10 + x*10) : (self.iter_ *len(self.users)*10+x*10+10)]
				tempList =[]
				for i in range(self.poolarticleSize):
					tempList.append(self.articles[randint(0,len(self.articles)-1)]) 
				self.articlePool = tempList
				#print 'poolSize', len(self.articlePool)
				
				#print 'user x', x	
				for alg_name, alg in algorithms.items():

					#print 'user x', self.users[x].id, 'true Theta', self.users[x].theta

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleLinUCB(self.users[x], self.articlePool)
					#print 'optimalArticle', self.users[x].id, '  ',  optimalArticle.id
					# The actually chosen article for user x at time t  
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)
					#print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardLinUCB(self.users[x], optimalArticle)
					#print OptimalReward[x]
					TempAccOptimalReward[x] +=OptimalReward[x]
					#print TempAccOptimalReward[x]
					#print 'optimalArticleReward', optimalArticle.id, OptimalReward
					# Compute the actually reward received
					click[x] = self.getExpectedRewardLinUCB( self.users[x], pickedArticle)
					TempAccReward[x] += click[x]

					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccOptimalReward[x] - TempAccReward[x] )				
					if click < 0:
						print click
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			#print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))
			
			with open(filenameWriteReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedReward[i][-1] ) for i in range(self.n_users)]))
				f.write(',' + str(AverageReward[-1]))
				f.write('\n')
			with open(filenameWriteOptimalReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedOptimalReward[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageOptimalReward[-1]))
				f.write('\n')
			with open(filenameWriteRegret, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedRegret[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageRegret[-1]))
				f.write('\n')
			#print 'InstantRegret', R[self.iter_]

			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				print BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				print BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				print BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 

		plt.plot(tim_, BatchAverageRegret, label = 'UCB1_Regret')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")
		

	def runGreedy(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'Greedy'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')

		
		with open(filenameWriteReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccReward')
			f.write('\n')
		with open(filenameWriteOptimalReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccOptimalReward')
			f.write('\n')
		with open(filenameWriteRegret, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'Regret')
			f.write('\n')
			

		tim_ = []
		AccumulatedReward= {}   # dictionary of AccumulatedReward array
		AccumulatedOptimalReward = {}
		AccumulatedRegret = {}
		TempAccReward = {}       # dictionary of currentAccumulatedReward
		TempAccOptimalReward = {} 
		AverageReward =[]
		AverageOptimalReward = []
		AverageRegret = []
		OptimalReward = {}
		click = {}
		#Initialize Reward and OptimalReward
		for i in range(self.n_users):
			AccumulatedReward[i] = []
			AccumulatedOptimalReward[i] = []
			AccumulatedRegret[i] = []
			TempAccReward[i] = 0.0
			TempAccOptimalReward[i] = 0.0
			OptimalReward[i] = 0.0
			click[i] = 0.0 
			

		# About Batch
		BatchAverageReward = []
		BatchAverageOptimalReward = []
		BatchAverageRegret = []

		for self.iter_ in xrange(self.iterations):
			#Generate users by time
			#self.regulateEnvironment()     
			
			#print math.floor(2.3)
			#x = int(math.floor(random()*10))    # generate a userID between 0-10 through random sampling
			#print x

			
			
			for x in self.users:

				#print 'Total Article', len(self.articles)

				#self.articlePool = self.articles[(self.iter_ *len(self.users)*10 + x*10) : (self.iter_ *len(self.users)*10+x*10+10)]
				tempList =[]
				for i in range(self.poolarticleSize):
					tempList.append(self.articles[randint(0,len(self.articles)-1)]) 
				self.articlePool = tempList
				#print 'poolSize', len(self.articlePool)
				
				#print 'user x', x	
				for alg_name, alg in algorithms.items():

					#print 'user x', self.users[x].id, 'true Theta', self.users[x].theta

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleLinUCB(self.users[x], self.articlePool)
					#print 'optimalArticle', self.users[x].id, '  ',  optimalArticle.id
					# The actually chosen article for user x at time t  
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)
					#print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardLinUCB(self.users[x], optimalArticle)
					#print OptimalReward[x]
					TempAccOptimalReward[x] +=OptimalReward[x]
					#print TempAccOptimalReward[x]
					#print 'optimalArticleReward', optimalArticle.id, OptimalReward
					# Compute the actually reward received
					click[x] = self.getExpectedRewardLinUCB( self.users[x], pickedArticle)
					TempAccReward[x] += click[x]

					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccOptimalReward[x] - TempAccReward[x] )				
					if click < 0:
						print click
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))
			
			with open(filenameWriteReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedReward[i][-1] ) for i in range(self.n_users)]))
				f.write(',' + str(AverageReward[-1]))
				f.write('\n')
			with open(filenameWriteOptimalReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedOptimalReward[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageOptimalReward[-1]))
				f.write('\n')
			with open(filenameWriteRegret, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedRegret[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageRegret[-1]))
				f.write('\n')
			#print 'InstantRegret', R[self.iter_]

			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				print BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				print BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				print BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 

		plt.plot(tim_, BatchAverageRegret, label = 'e-Greedy_Regret')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")


	def runUCB1_Multi(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'Multi_UCB1'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')

		
		with open(filenameWriteReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccReward')
			f.write('\n')
		with open(filenameWriteOptimalReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccOptimalReward')
			f.write('\n')
		with open(filenameWriteRegret, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'Regret')
			f.write('\n')
			

		tim_ = []
		AccumulatedReward= {}   # dictionary of AccumulatedReward array
		AccumulatedOptimalReward = {}
		AccumulatedRegret = {}
		TempAccReward = {}       # dictionary of currentAccumulatedReward
		TempAccOptimalReward = {} 
		AverageReward =[]
		AverageOptimalReward = []
		AverageRegret = []
		OptimalReward = {}
		click = {}
		#Initialize Reward and OptimalReward
		for i in range(self.n_users):
			AccumulatedReward[i] = []
			AccumulatedOptimalReward[i] = []
			AccumulatedRegret[i] = []
			TempAccReward[i] = 0.0
			TempAccOptimalReward[i] = 0.0
			OptimalReward[i] = 0.0
			click[i] = 0.0 
			

		# About Batch
		BatchAverageReward = []
		BatchAverageOptimalReward = []
		BatchAverageRegret = []

		for self.iter_ in xrange(self.iterations):
			#Generate users by time
			#self.regulateEnvironment()     
			
			#print math.floor(2.3)
			#x = int(math.floor(random()*10))    # generate a userID between 0-10 through random sampling
			#print x

			
			
			for x in self.users:
				#print 'Total Article', len(self.articles)

				#self.articlePool = self.articles[(self.iter_ *len(self.users)*10 + x*10) : (self.iter_ *len(self.users)*10+x*10+10)]
				tempList =[]
				for i in range(self.poolarticleSize):
					tempList.append(self.articles[randint(0,len(self.articles)-1)]) 
				self.articlePool = tempList
				#print 'poolSize', len(self.articlePool)
				
				#print 'user x', x	
				for alg_name, alg in algorithms.items():

					#print 'user x', self.users[x].id, 'true Theta', self.users[x].theta

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleLinUCB(self.users[x], self.articlePool)
					#print 'optimalArticle', self.users[x].id, '  ',  optimalArticle.id
					# The actually chosen article for user x at time t  
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)
					#print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardLinUCB(self.users[x], optimalArticle)
					#print OptimalReward[x]
					TempAccOptimalReward[x] +=OptimalReward[x]
					#print TempAccOptimalReward[x]
					#print 'optimalArticleReward', optimalArticle.id, OptimalReward
					# Compute the actually reward received
					click[x] = self.getExpectedRewardLinUCB( self.users[x], pickedArticle)
					TempAccReward[x] += click[x]

					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccOptimalReward[x] - TempAccReward[x] )				
					if click < 0:
						print click
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))
			
			with open(filenameWriteReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedReward[i][-1] ) for i in range(self.n_users)]))
				f.write(',' + str(AverageReward[-1]))
				f.write('\n')
			with open(filenameWriteOptimalReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedOptimalReward[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageOptimalReward[-1]))
				f.write('\n')
			with open(filenameWriteRegret, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedRegret[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageRegret[-1]))
				f.write('\n')
			#print 'InstantRegret', R[self.iter_]

			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				print BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				print BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				print BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 

		plt.plot(tim_, BatchAverageRegret, label = 'M_UCB1')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")

		'''		
		f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = False)
		#ax1.plot(tim_, BatchAverageRegret, label = 'AverageRegret')
		ax1.plot(tim_, BatchAverageReward, label = 'AverageReward')
		ax1.plot(tim_, BatchAverageOptimalReward, label = 'AverageOptimalReward')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Average Regret')
		ax1.set_title('Collaborative LinUCB')
		ax1.legend()
		ax2.plot()
		ax2.set_xlabel('time')
		ax2.set_ylabel('Average Regret Ratio')

		'''

	def runGreedy_Multi(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'Multi_Greedy'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')

		
		with open(filenameWriteReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccReward')
			f.write('\n')
		with open(filenameWriteOptimalReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccOptimalReward')
			f.write('\n')
		with open(filenameWriteRegret, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'Regret')
			f.write('\n')
			

		tim_ = []
		AccumulatedReward= {}   # dictionary of AccumulatedReward array
		AccumulatedOptimalReward = {}
		AccumulatedRegret = {}
		TempAccReward = {}       # dictionary of currentAccumulatedReward
		TempAccOptimalReward = {} 
		AverageReward =[]
		AverageOptimalReward = []
		AverageRegret = []
		OptimalReward = {}
		click = {}
		#Initialize Reward and OptimalReward
		for i in range(self.n_users):
			AccumulatedReward[i] = []
			AccumulatedOptimalReward[i] = []
			AccumulatedRegret[i] = []
			TempAccReward[i] = 0.0
			TempAccOptimalReward[i] = 0.0
			OptimalReward[i] = 0.0
			click[i] = 0.0 
			

		# About Batch
		BatchAverageReward = []
		BatchAverageOptimalReward = []
		BatchAverageRegret = []

		for self.iter_ in xrange(self.iterations):
			#Generate users by time
			#self.regulateEnvironment()     
			
			#print math.floor(2.3)
			#x = int(math.floor(random()*10))    # generate a userID between 0-10 through random sampling
			#print x

			
			
			for x in self.users:

				#print 'Total Article', len(self.articles)

				#self.articlePool = self.articles[(self.iter_ *len(self.users)*10 + x*10) : (self.iter_ *len(self.users)*10+x*10+10)]
				tempList =[]
				for i in range(self.poolarticleSize):
					tempList.append(self.articles[randint(0,len(self.articles)-1)]) 
				self.articlePool = tempList
				#print 'poolSize', len(self.articlePool)
				
				#print 'user x', x	
				for alg_name, alg in algorithms.items():

					#print 'user x', self.users[x].id, 'true Theta', self.users[x].theta

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleLinUCB(self.users[x], self.articlePool)
					#print 'optimalArticle', self.users[x].id, '  ',  optimalArticle.id
					# The actually chosen article for user x at time t  
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)
					#print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardLinUCB(self.users[x], optimalArticle)
					#print OptimalReward[x]
					TempAccOptimalReward[x] +=OptimalReward[x]
					#print TempAccOptimalReward[x]
					#print 'optimalArticleReward', optimalArticle.id, OptimalReward
					# Compute the actually reward received
					click[x] = self.getExpectedRewardLinUCB( self.users[x], pickedArticle)
					TempAccReward[x] += click[x]

					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccOptimalReward[x] - TempAccReward[x] )				
					if click < 0:
						print click
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))
			
			with open(filenameWriteReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedReward[i][-1] ) for i in range(self.n_users)]))
				f.write(',' + str(AverageReward[-1]))
				f.write('\n')
			with open(filenameWriteOptimalReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedOptimalReward[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageOptimalReward[-1]))
				f.write('\n')
			with open(filenameWriteRegret, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedRegret[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageRegret[-1]))
				f.write('\n')
			#print 'InstantRegret', R[self.iter_]

			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				print BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				print BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				print BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 

		plt.plot(tim_, BatchAverageRegret, label = 'M_Greedy')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")

		'''		
		f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = False)
		#ax1.plot(tim_, BatchAverageRegret, label = 'AverageRegret')
		ax1.plot(tim_, BatchAverageReward, label = 'AverageReward')
		ax1.plot(tim_, BatchAverageOptimalReward, label = 'AverageOptimalReward')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Average Regret')
		ax1.set_title('Collaborative LinUCB')
		ax1.legend()
		ax2.plot()
		ax2.set_xlabel('time')
		ax2.set_ylabel('Average Regret Ratio')
		'''


	def runExp3_Multi(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'Multi_Exp3'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')

		
		with open(filenameWriteReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccReward')
			f.write('\n')
		with open(filenameWriteOptimalReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccOptimalReward')
			f.write('\n')
		with open(filenameWriteRegret, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'Regret')
			f.write('\n')
			

		tim_ = []
		AccumulatedReward= {}   # dictionary of AccumulatedReward array
		AccumulatedOptimalReward = {}
		AccumulatedRegret = {}
		TempAccReward = {}       # dictionary of currentAccumulatedReward
		TempAccOptimalReward = {} 
		AverageReward =[]
		AverageOptimalReward = []
		AverageRegret = []
		OptimalReward = {}
		click = {}
		#Initialize Reward and OptimalReward
		for i in range(self.n_users):
			AccumulatedReward[i] = []
			AccumulatedOptimalReward[i] = []
			AccumulatedRegret[i] = []
			TempAccReward[i] = 0.0
			TempAccOptimalReward[i] = 0.0
			OptimalReward[i] = 0.0
			click[i] = 0.0 
			

		# About Batch
		BatchAverageReward = []
		BatchAverageOptimalReward = []
		BatchAverageRegret = []

		for self.iter_ in xrange(self.iterations):
			#Generate users by time
			#self.regulateEnvironment()     
			
			#print math.floor(2.3)
			#x = int(math.floor(random()*10))    # generate a userID between 0-10 through random sampling
			#print x

			
			
			for x in self.users:

				#print 'Total Article', len(self.articles)

				#self.articlePool = self.articles[(self.iter_ *len(self.users)*10 + x*10) : (self.iter_ *len(self.users)*10+x*10+10)]
				tempList =[]
				for i in range(self.poolarticleSize):
					tempList.append(self.articles[randint(0,len(self.articles)-1)]) 
				self.articlePool = tempList
				#print 'poolSize', len(self.articlePool)
				
				#print 'user x', x	
				for alg_name, alg in algorithms.items():

					#print 'user x', self.users[x].id, 'true Theta', self.users[x].theta

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleLinUCB(self.users[x], self.articlePool)
					#print 'optimalArticle', self.users[x].id, '  ',  optimalArticle.id
					# The actually chosen article for user x at time t  
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)
					#print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardLinUCB(self.users[x], optimalArticle)
					#print OptimalReward[x]
					TempAccOptimalReward[x] +=OptimalReward[x]
					#print TempAccOptimalReward[x]
					#print 'optimalArticleReward', optimalArticle.id, OptimalReward
					# Compute the actually reward received
					click[x] = self.getExpectedRewardLinUCB( self.users[x], pickedArticle)
					TempAccReward[x] += click[x]

					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccOptimalReward[x] - TempAccReward[x] )				
					if click < 0:
						print click
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))
			
			with open(filenameWriteReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedReward[i][-1] ) for i in range(self.n_users)]))
				f.write(',' + str(AverageReward[-1]))
				f.write('\n')
			with open(filenameWriteOptimalReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedOptimalReward[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageOptimalReward[-1]))
				f.write('\n')
			with open(filenameWriteRegret, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedRegret[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageRegret[-1]))
				f.write('\n')
			#print 'InstantRegret', R[self.iter_]

			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				print BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				print BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				print BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 

		plt.plot(tim_, BatchAverageRegret, label = 'M_Exp3')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")

		'''		
		f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = False)
		#ax1.plot(tim_, BatchAverageRegret, label = 'AverageRegret')
		ax1.plot(tim_, BatchAverageReward, label = 'AverageReward')
		ax1.plot(tim_, BatchAverageOptimalReward, label = 'AverageOptimalReward')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Average Regret')
		ax1.set_title('Collaborative LinUCB')
		ax1.legend()
		ax2.plot()
		ax2.set_xlabel('time')
		ax2.set_ylabel('Average Regret Ratio')
		'''




	def runLinUCB(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'Revised_LinUCB'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')

		
		with open(filenameWriteReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccReward')
			f.write('\n')
		with open(filenameWriteOptimalReward, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'AccOptimalReward')
			f.write('\n')
		with open(filenameWriteRegret, 'a+') as f:
			f.write('Time(Iteration)')
			f.write(',' + 'Regret')
			f.write('\n')
			

		tim_ = []
		iter_ = []
		AccumulatedReward= {}   # dictionary of AccumulatedReward array
		AccumulatedOptimalReward = {}
		AccumulatedRegret = {}
		TempAccReward = {}       # dictionary of currentAccumulatedReward
		TempAccOptimalReward = {} 
		TempAccRegret  = {}
		AverageReward =[]
		AverageOptimalReward = []
		AverageRegret = []
		OptimalReward = {}
		click = {}
		Regret = {}
		#Initialize Reward and OptimalReward
		for i in range(self.n_users):
			AccumulatedReward[i] = []
			AccumulatedOptimalReward[i] = []
			AccumulatedRegret[i] = []
			TempAccReward[i] = 0.0
			TempAccOptimalReward[i] = 0.0
			TempAccRegret[i] = 0.0
			OptimalReward[i] = 0.0
			click[i] = 0.0 
			Regret[i] =0.0
			

		# About Batch
		BatchAverageReward = []
		BatchAverageOptimalReward = []
		BatchAverageRegret = []

		for self.iter_ in xrange(self.iterations):
			iter_.append(self.iter_)

			#Generate users by time
			#self.regulateEnvironment()     
			
			#print math.floor(2.3)
			#x = int(math.floor(random()*10))    # generate a userID between 0-10 through random sampling
			#print x

			
			
			for x in self.users:

				#print 'Total Article', len(self.articles)

				#self.articlePool = self.articles[(self.iter_ *len(self.users)*10 + x*10) : (self.iter_ *len(self.users)*10+x*10+10)]
				tempList =[]
				for i in range(self.poolarticleSize):
					tempList.append(self.articles[randint(0,len(self.articles)-1)]) 
				self.articlePool = tempList
				#print 'poolSize', len(self.articlePool)
				
				#print 'user x', x	
				for alg_name, alg in algorithms.items():

					#print 'user x', self.users[x].id, 'true Theta', self.users[x].theta

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleLinUCB(self.users[x], self.articlePool)
					#print 'optimalArticle', self.users[x].id, '  ',  optimalArticle.id
					# The actually chosen article for user x at time t  
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)
					#print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardLinUCB(self.users[x], optimalArticle)
					#print OptimalReward[x]
					TempAccOptimalReward[x] +=OptimalReward[x]
					#print TempAccOptimalReward[x]
					#print 'optimalArticleReward', optimalArticle.id, OptimalReward
					# Compute the actually reward received
					click[x] = self.getExpectedRewardLinUCB( self.users[x], pickedArticle)
					Regret[x] = OptimalReward[x] - click[x]
					TempAccRegret[x] +=Regret[x]
					TempAccReward[x] += click[x]

					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccRegret[x])				
					if click < 0:
						print click
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))
			
			with open(filenameWriteReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedReward[i][-1] ) for i in range(self.n_users)]))
				f.write(',' + str(AverageReward[-1]))
				f.write('\n')
			with open(filenameWriteOptimalReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedOptimalReward[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageOptimalReward[-1]))
				f.write('\n')
			with open(filenameWriteRegret, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedRegret[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageRegret[-1]))
				f.write('\n')
			#print 'InstantRegret', R[self.iter_]

			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 

		plt.plot(tim_, BatchAverageRegret, label = 'LinUCB_Regret')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")

		plt.plot(self.iter_, AccumulatedRegret[0])

		'''		
		f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = False)
		#ax1.plot(tim_, BatchAverageRegret, label = 'AverageRegret')
		ax1.plot(tim_, BatchAverageReward, label = 'AverageReward')
		ax1.plot(tim_, BatchAverageOptimalReward, label = 'AverageOptimalReward')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Average Regret')
		ax1.set_title('Collaborative LinUCB')
		ax1.legend()
		ax2.plot()
		ax2.set_xlabel('time')
		ax2.set_ylabel('Average Regret Ratio')
		'''


		def plotLines(axes_, xlocs):
			for xloc, color in xlocs:
				# axes = plt.gca()
				for x in xloc:
					xSet = [x for _ in range(31)]
					ymin, ymax = axes_.get_ylim()
					ySet = ymin + (np.array(range(0, 31))*1.0/30) * (ymax - ymin)
					axes_.plot(xSet, ySet, color)

		xlocs = [(list(set(map(lambda x: x.startTime, self.articles))), "black")]
		#plotLines(ax1, xlocs)
		#plotLines(ax2, xlocs)
		


	def runCoLinUCB(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'Revised_CoUCB'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')

		with open(filenameWriteReward, 'a+') as f:
			f.write('Time(Iteration)'+',' + ','.join([str(self.users[i].id) for i in range(self.n_users)]))
			f.write(',' + 'AverageAccReward')
			f.write('\n')
		with open(filenameWriteOptimalReward, 'a+') as f:
			f.write('Time(Iteration)'+',' + ','.join([str(self.users[i].id) for i in range(self.n_users)]))
			f.write(',' + 'AverageAccOptimalReward')
			f.write('\n')
		with open(filenameWriteRegret, 'a+') as f:
			f.write('Time(Iteration)'+',' + ','.join([str(self.users[i].id) for i in range(self.n_users)]))
			f.write(',' + 'AverageRegret')
			f.write('\n')
		tim_ = []
		AccumulatedReward= {}   # dictionary of AccumulatedReward array
		AccumulatedOptimalReward = {}
		AccumulatedRegret = {}
		TempAccReward = {}       # dictionary of currentAccumulatedReward
		TempAccOptimalReward = {} 
		TempAccRegret  = {}
		AverageReward =[]
		AverageOptimalReward = []
		AverageRegret = []
		OptimalReward = {}
		click = {}
		Regret = {}
		#Initialize Reward and OptimalReward
		for i in range(self.n_users):
			AccumulatedReward[i] = []
			AccumulatedOptimalReward[i] = []
			AccumulatedRegret[i] = []
			TempAccReward[i] = 0.0
			TempAccOptimalReward[i] = 0.0
			TempAccRegret[i] = 0.0
			OptimalReward[i] = 0.0
			click[i] = 0.0 
			Regret[i] =0.0
			

		# About Batch
		BatchAverageReward = []
		BatchAverageOptimalReward = []
		BatchAverageRegret = []
		for self.iter_ in xrange(self.iterations):
			#self.regulateEnvironment()
			
			for x in self.users:
				#print 'Total Article', len(self.articles)

				#self.articlePool = self.articles[(self.iter_ *len(self.users)*10 + x*10) : (self.iter_ *len(self.users)*10+x*10+10)]
				#tempList = []
				tempList =[]
				for i in range(self.poolarticleSize):
					tempList.append(self.articles[randint(0,len(self.articles)-1)]) 
				self.articlePool = tempList
				#print 'poolSize', len(self.articlePool)

				#print 'user x', x	
				for alg_name, alg in algorithms.items():

					#print 'user x', self.users[x].id, 'true Theta', self.users[x].theta

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleCoLinUCB(self.users[x], self.articlePool)
					print 'optimalArticle', self.users[x].id, '  ',  optimalArticle.id
					print 'featureveator:', optimalArticle.featureVector
					# The actually chosen article for user x at time t  
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)
					print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardCoLinUCB(self.users[x], optimalArticle)
					print OptimalReward[x]
					TempAccOptimalReward[x] +=OptimalReward[x]
					print TempAccOptimalReward[x]
					#print 'optimalArticleReward', optimalArticle.id, OptimalReward
					# Compute the actually reward received
					click[x] = self.getExpectedRewardCoLinUCB( self.users[x], pickedArticle)
					Regret[x] = OptimalReward[x] - click[x]
					TempAccReward[x] += click[x]
					TempAccRegret[x] +=Regret[x]


					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccRegret[x])				
					if click < 0:
						print click
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))
			
			with open(filenameWriteReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedReward[i][-1] ) for i in range(self.n_users)]))
				f.write(',' + str(AverageReward[-1]))
				f.write('\n')
			with open(filenameWriteOptimalReward, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedOptimalReward[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageOptimalReward[-1]))
				f.write('\n')
			with open(filenameWriteRegret, 'a+') as f:
				f.write(str(self.iter_))
				f.write(',' + ','.join([str(AccumulatedRegret[i][-1]) for i in range(self.n_users)]))
				f.write(',' + str(AverageRegret[-1]))
				f.write('\n')
			#print 'InstantRegret', R[self.iter_]

			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 

		plt.plot(tim_, BatchAverageRegret, label = 'CoLinUCB_Regert')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")
		'''		
		f, (ax1, ax2) = plt.subplots(2, sharex = True, sharey = False)
		#ax1.plot(tim_, BatchAverageRegret, label = 'AverageRegret')
		ax1.plot(tim_, BatchAverageReward, label = 'AverageReward')
		ax1.plot(tim_, BatchAverageOptimalReward, label = 'AverageOptimalReward')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Average Regret')
		ax1.set_title('Collaborative LinUCB')
		ax1.legend()
		ax2.plot()
		ax2.set_xlabel('time')
		ax2.set_ylabel('Average Regret Ratio')
		'''


		def plotLines(axes_, xlocs):
			for xloc, color in xlocs:
				# axes = plt.gca()
				for x in xloc:
					xSet = [x for _ in range(31)]
					ymin, ymax = axes_.get_ylim()
					ySet = ymin + (np.array(range(0, 31))*1.0/30) * (ymax - ymin)
					axes_.plot(xSet, ySet, color)

		xlocs = [(list(set(map(lambda x: x.startTime, self.articles))), "black")]
		#plotLines(ax1, xlocs)
		#plotLines(ax2, xlocs)
		

	def batchRecord(self, userArrived, algorithms):
		'''
		for alg_name, alg in algorithms.items():
			poolArticlesCTR = dict([(x.id, alg.getarticleCTR(userArrived.id, x.id)) for x in self.articlePool])
			if self.iter_%self.batchSize == 0:
				self.alg_perf[alg_name].addRecord(self.iter_, self.getPoolMSE(alg), poolArticlesCTR)
			
			for article in self.articlePool:
				article.addRecord(self.iter_, self.getArticleAbsDiff(alg, userArrived), alg_name)
			'''
		print "Iteration %d"%self.iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime



if __name__ == '__main__':
	iterations = 1000
	dimension =5
	alpha  = .3 
	lambda_ = 0.5   # Inialize A
	gamma = .4  # parameter in Exp3
	epsilon = .2  # parameter in Epsilon greedy

	n_articles = 10000
	n_users = 10
	poolArticles = [10]
	articleInflux = 10
	batchSize = 5

	userFilename = os.path.join(sim_files_folder, "users+it-"+str(iterations)+"+dim-"+str(dimension)+".p")
	resultsFile = os.path.join(result_folder, "Results.csv")

	"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
	UM = UserManager(dimension, iterations, userFilename, thetaFunc=gaussianFeature)
	#UM.simulateThetafromUsers(n_users, thetaFunc=gaussianFeature,  argv={'l2_limit':1})
	

	for p_art in poolArticles:
		articlesFilename = os.path.join(sim_files_folder, "articles"+str(n_articles)+"+AP-"+str(p_art)+"+influx"+str(articleInflux)+"+IT-"+str(iterations)+"+dim"+str(dimension)+".p")
		AM = ArticleManager(iterations, dimension, n_articles=n_articles, 
				poolArticles=p_art, thetaFunc=featureUniform,  argv={'l2_limit':1},
				influx=articleInflux)
		#articles = AM.simulateArticlePool()
		#AM.saveArticles(articles, articlesFilename, force=False)
		
		#print map(lambda x:x.startTime, articles), map(lambda x:x.endTime, articles)

		UM = UserManager(dimension, iterations, userFilename, thetaFunc=featureUniform)
		articles = AM.loadArticles(articlesFilename)

		simExperiment = simulateOnlineData(dimension  = dimension,
							iterations = iterations,
							articles=articles,
							n_users = n_users,
							userGenerator = UM.ThetaIterator(),	
							userGenerator1 = UM.ThetaIterator(),		
							noise = lambda : 0,
							batchSize = batchSize,
							type_ = "ConstantTheta",environmentVars={},
							signature = AM.signature,
							poolarticleSize = p_art
					)
		print "Starting for ", simExperiment.simulation_signature
		algorithmsCoLinUCB = {}
		algorithmsLinUCB = {}
		algorithmsExp3 ={}
		algorithmsUCB1 = {}
		algorithmsGreedy = {}
		algorithmsExp3_Multi = {}
		algorithmsUCB1_Multi = {}
		algorithmsGreedy_Multi = {}

              	
		algorithmsCoLinUCB["CoLinUCB"] = CoLinUCBAlgorithm(dimension=dimension, alpha=alpha, lambda_ = lambda_, n = n_users)
		algorithmsLinUCB['LinUCB'] = LinUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
		#algorithmsExp3['Exp3'] = Exp3Algorithm(dimension = dimension, gamma =gamma)
		#algorithmsUCB1['UCB1'] = UCB1Algorithm(dimension = dimension)
		#algorithmsGreedy['epsilon-Greedy'] = EpsilonGreedyAlgorithm(dimension = dimension, epsilon = epsilon)
		#algorithmsExp3_Multi['exp3_multi'] = Exp3_Multi_Algorithm(dimension= dimension, gamma = gamma, n = n_users)
		#algorithmsUCB1_Multi['UCB1_multi'] = UCB1_Multi_Algorithm(dimension = dimension, n = n_users)
		#algorithmsGreedy_Multi['Greedy_multi'] = EpsilonGreedy_Multi_Algorithm(dimension =dimension, epsilon = epsilon, n = n_users)
		
		
		simExperiment.runCoLinUCB(algorithmsCoLinUCB)
		simExperiment.runLinUCB(algorithmsLinUCB)
		#simExperiment.runExp3(algorithmsExp3)
		#simExperiment.runUCB1(algorithmsUCB1)
		#simExperiment.runGreedy(algorithmsGreedy)
		#simExperiment.runExp3_Multi(algorithmsExp3_Multi)
		#simExperiment.runUCB1_Multi(algorithmsUCB1_Multi)
		#simExperiment.runGreedy_Multi(algorithmsGreedy_Multi)
		

	
