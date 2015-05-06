import math
import numpy as np 

#from CoLinUCB import *
#from LinUCB import *
from Algori import *
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
		epsilon = 0.4
		I = np.identity(n)
		W = I - epsilon * L  # W is a double stostastic matrix
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

	def CoTheta(self):
		tempW = self.W
		for i in range(self.n_users):
			temp1 = np.zeros(self.dimension)
			for j in range(self.n_users):
				temp1 += tempW[self.users[j].id][i] * np.asarray(self.users[j].theta)
			self.users[i].CoTheta = temp1

	def getExpectedRewardCoLinUCB(self, UserArrived, pickedArticle):
		reward = np.dot(UserArrived.CoTheta, pickedArticle.featureVector)
		return reward
  
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


	def runLinUCB(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'Revised_LinUCB'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')
			

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
			wrong = 0
			for x in self.users:

				# Randomly choose 10 articles
				tempList =[]
				for i in range(self.poolarticleSize):
					tempList.append(self.articles[randint(0,len(self.articles)-1)]) 
				self.articlePool = tempList

				for alg_name, alg in algorithms.items():

					#print 'user x', self.users[x].id, 'true Theta', self.users[x].theta

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleCoLinUCB(self.users[x], self.articlePool)
					#print 'optimalArticle', self.users[x].id, '  ',  optimalArticle.id
					# The actually chosen article for user x at time t  
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)

					if optimalArticle !=pickedArticle:
						wrong +=1
					#print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardCoLinUCB(self.users[x], optimalArticle)
					#print OptimalReward[x]
					TempAccOptimalReward[x] +=OptimalReward[x]
					#print TempAccOptimalReward[x]
					#print 'optimalArticleReward', optimalArticle.id, OptimalReward
					# Compute the actually reward received
					click[x] = self.getExpectedRewardCoLinUCB( self.users[x], pickedArticle)
					Regret[x] = OptimalReward[x] - click[x]
					print Regret[x]
					TempAccRegret[x] +=Regret[x]
					TempAccReward[x] += click[x]

					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccRegret[x])				
					'''
					if int(x) == 1:
						print 'user', x, 'True-useTheta', self.users[x].theta
					'''
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)

				#print 'wrongNum', wrong

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			#print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))


			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 
		print "Length", AccumulatedRegret[0]
		plt.plot(tim_, BatchAverageRegret, label = 'LinUCB_Regret')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")

		#plt.plot(self.iter_, AccumulatedRegret[0])

	def runCoLinUCB(self, algorithms):
		self.CoTheta()
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = 'Revised_CoUCB'
		filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
		filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
		filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')

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
			wrong = 0
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

					#get optimal Article for user x at time t
					optimalArticle = self.GetOptimalArticleCoLinUCB(self.users[x], self.articlePool)
					pickedArticle = alg.decide(self.articlePool, self.users[x].id, self.iter_)

					if optimalArticle != pickedArticle:
						wrong +=1
					#print 'pickedArticle', pickedArticle.id
					# Compute the optimal reward
					OptimalReward[x] = self.getExpectedRewardCoLinUCB(self.users[x], optimalArticle)
					# Compute the actually reward received
					click[x] = self.getExpectedRewardCoLinUCB( self.users[x], pickedArticle)
					#click[x] = self.getExpectedRewardLinUCB( self.users[x], pickedArticle)
					Regret[x] = OptimalReward[x] - click[x]
					#print Regret[x]

					TempAccOptimalReward[x] +=OptimalReward[x]
					TempAccReward[x] += click[x]
					TempAccRegret[x] +=Regret[x]


					AccumulatedOptimalReward[x].append(TempAccOptimalReward[x])
					AccumulatedReward[x].append(TempAccReward[x])	
					AccumulatedRegret[x].append(TempAccRegret[x])				
					if click < 0:
						print click
					'''
					if int(x) == 1:
						print 'user', x, 'True-useCoTheta', self.users[x].theta
					'''
					
					alg.updateParameters(pickedArticle, click[x], self.users[x].id)
				#print'wrongNum', wrong

					#self.iterationRecord(alg_name, self.users[x].id, click, pickedArticle.id)  ###Change
			AverageReward.append((sum([AccumulatedReward[i][-1] for i in range(self.n_users)]) / self.n_users))
			AverageOptimalReward.append((sum([AccumulatedOptimalReward[i][-1] for i in range(self.n_users)] )/ self.n_users))
			#print AverageOptimalReward[-1]
			AverageRegret.append((sum([AccumulatedRegret[i][-1] for i in range(self.n_users)]) / self.n_users))

			#print 'InstantRegret', R[self.iter_]
			#print self.iter_
			if self.iter_%self.batchSize ==0 and self.iter_ >1:
				self.batchRecord(self.users[x], algorithms)
				tim_.append(self.iter_)
				BatchAverageReward.append(sum(AverageReward) / (1.0* self.batchSize))
				BatchAverageOptimalReward.append(sum(AverageOptimalReward) / (1.0* self.batchSize))
				BatchAverageRegret.append(sum(AverageRegret) / (1.0* self.batchSize))
				AverageReward = []
				AverageOptimalReward =[]
				AverageRegret = [] 

		print 'CoLinUCBLength', AccumulatedRegret[0]
		plt.plot(tim_, BatchAverageRegret, label = 'CoLinUCB_Regert')
		plt.legend()
		plt.xlabel("Iteration")
		plt.ylabel("Regret")

		

	def batchRecord(self, userArrived, algorithms):
		print "Iteration %d"%self.iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime



if __name__ == '__main__':
	iterations = 1000
	dimension =5
	alpha  = .3 
	lambda_ = 0.5   # Inialize A
	gamma = .4  # parameter inc Exp3
	epsilon = .2  # parameter in Epsilon greedy

	n_articles = 1000
	n_users = 10
	poolArticles = [10]
	articleInflux = 0
	batchSize = 5

	userFilename = os.path.join(sim_files_folder, "users+it-"+str(iterations)+"+dim-"+str(dimension)+".p")
	resultsFile = os.path.join(result_folder, "Results.csv")

	"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
	UM = UserManager(dimension, iterations, userFilename, thetaFunc=gaussianFeature)
	#UM.simulateThetafromUsers(n_users, thetaFunc=featureUniform,  argv={'l2_limit':1})
	

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
		#algorithmsLinUCB = {}
		       	
		algorithmsCoLinUCB["CoLinUCB"] = CoLinUCBAlgorithm(dimension=dimension, alpha=alpha, lambda_ = lambda_, n = n_users)
		algorithmsLinUCB['LinUCB'] = LinUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
		
		simExperiment.runCoLinUCB(algorithmsCoLinUCB)
		simExperiment.runLinUCB(algorithmsLinUCB)
		
	
