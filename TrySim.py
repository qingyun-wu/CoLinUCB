import math
import numpy as np 

#from CoLinUCB import *
#from LinUCB import *
#from Algori import *

#from DiagAlgori import *
from TryAlg import *
#from CoAlg import *
#from exp3_MAB import *
#from exp3_MultiUser import *
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
	def __init__(self, dimension, iterations, articles, users, 
		batchSize = 1000,
		noise = lambda : 0,
		type_ = 'UniformTheta', environmentVars = None,
		signature = '', poolarticleSize = 10):

		self.simulation_signature = signature
		self.type = type_
		self.environmentVars = environmentVars

		self.dimension = dimension
		self.iterations = iterations
		self.noise = noise
		self.articles = articles 
		self.users = users

		self.poolarticleSize = poolarticleSize
		self.batchSize = batchSize
		
		self.W = self.initializeW()
		self.initiateEnvironment()
	

	def initializeW(self):
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
		W = I - epsilon * L  # W is a double stostastic matrix
		return W

	def getW(self):
		return self.W

	def batchRecord(self, iter_):
		print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime


	def initiateEnvironment(self):
		env_sign = self.type
		sig_name = [("It",str(self.iterations//1000)+'k')]
		sig_name = '_'.join(['-'.join(tup) for tup in sig_name])
		self.simulation_signature += '_' + env_sign + '_' + sig_name

	def regulateArticlePool(self):
		tempArticlePool =[]
		for i in range(self.poolarticleSize):
			tempArticlePool.append(self.articles[randint(0,len(self.articles)-1)]) 
		self.articlePool = tempArticlePool


	def CoTheta(self):
		tempW = self.W
		for i in range(len(self.users)):
			tempTheta = np.zeros(self.dimension)
			for j in range(len(self.users)):
				tempTheta += tempW[self.users[j].id][i] * np.asarray(self.users[j].theta)
			self.users[i].CoTheta = tempTheta
			print 'users', i, 'CoTheta', self.users[i].CoTheta

	def getReward(self, user, pickedArticle):
		reward = np.dot(user.CoTheta, pickedArticle.featureVector)
		return reward

	def GetOptimalArticle(self, user, articlePool):
		reward ={}
		for x in articlePool:	 
			reward[x.id] = self.getReward(user, x)
		optimalArticle = max([(x, reward[x.id]) for x in articlePool], key = itemgetter(1))[0]
		return optimalArticle
	def getThetaAbsDiff(self, user, alg):
		return np.linalg.norm(user.theta - alg.getLearntParameters(user.id))
	def getCoThetaAbsDiff(self, user, alg):	
		return np.linalg.norm(user.CoTheta - alg.getLearntParameters(user.id))


	def runAlgorithms(self, algorithms):
		self.CoTheta()
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 


		tim_ = {}
		UserAverageRegret = {}
		BatchAverageRegret = {}
		CurrentAccRegret = {}
		AccRegret = {}
		regret = {}
		pickedArticle = {}
		reward = {}
		# Iniatilization
		for alg_name, alg in algorithms.items():
			'''
			fileSig = str(alg_name)
			filenameWriteReward = os.path.join(save_address, fileSig+ '_AccReward' + timeRun + '.csv')
			filenameWriteOptimalReward = os.path.join(save_address, fileSig+ '_AccOptimalReward' + timeRun + '.csv')
			filenameWriteRegret = os.path.join(save_address, fileSig+ '_AccRegret' + timeRun + '.csv')
			'''
			tim_[alg_name] = []
			UserAverageRegret[alg_name] = []
			BatchAverageRegret[alg_name] = []
			CurrentAccRegret[alg_name] = {}
			AccRegret[alg_name] = {}
			regret[alg_name] = {}
			pickedArticle[alg_name] = {}
			reward[alg_name] = {}
			

			for i in range(len(self.users)):
				AccRegret[alg_name][i] = []
				CurrentAccRegret[alg_name][i] = 0.0
		ThetaDiffList = []
		CoThetaDiffList = []
		BatchThetaDiff = []
		BatchCoThetaDiff = []
		time = []
		# Loop begin
		for iter_ in range(self.iterations):
			ThetaDiff = 0
			CoThetaDiff = 0
			time.append(iter_)
			for x in  range(len(self.users)):

				self.regulateArticlePool()

				noise = self.noise()
				#get optimal Article for user x at time t
				optimalArticle = self.GetOptimalArticle(self.users[x], self.articlePool)
				OptimalReward = self.getReward(self.users[x], optimalArticle) + noise

				for alg_name, alg in algorithms.items():
					pickedArticle[alg_name][x] = alg.decide(self.articlePool, self.users[x].id)
					reward[alg_name][x] = self.getReward(self.users[x], pickedArticle[alg_name][x]) + noise

					alg.updateReward(pickedArticle[alg_name][x], reward[alg_name][x], self.users[x].id)
					
					regret[alg_name][x] = OptimalReward - reward[alg_name][x]

					CurrentAccRegret[alg_name][x] = CurrentAccRegret[alg_name][x]+regret[alg_name][x]
					AccRegret[alg_name][x].append(CurrentAccRegret[alg_name][x])
				ThetaDiff += self.getThetaAbsDiff(self.users[x], algorithms['CoLinUCB'])
				CoThetaDiff += self.getCoThetaAbsDiff(self.users[x], algorithms['LinUCB'])

			ThetaDiffList.append(ThetaDiff)
			CoThetaDiffList.append(CoThetaDiff)

			for alg_name, alg in algorithms.items():
				for x in range(len(self.users)):
					alg.updateParameters(self.users[x].id)

				UserAverageRegret[alg_name].append((sum([AccRegret[alg_name][i][-1] for i in range(len(users))]) / len(users)))
				if iter_%self.batchSize ==0 and iter_ >1:
					self.batchRecord(iter_)
					tim_[alg_name].append(iter_)
					BatchAverageRegret[alg_name].append(sum(UserAverageRegret[alg_name]) / (1.0* self.batchSize))
					#BatchAverageRegret[alg_name].append(sum(AccRegret[alg_name][1]) / (1.0* self.batchSize))
					UserAverageRegret[alg_name] = []
					#AccRegret[alg_name][1] = []

		f, axa = plt.subplots(2, sharex=True)

		for alg_name, alg in algorithms.items():		
			axa[0].plot(tim_[alg_name], BatchAverageRegret[alg_name], label = alg_name)
			axa[0].legend()
			axa[0].set_xlabel("Iteration")
			axa[0].set_ylabel("Regret")
			#plt.title("Noise scale = .0001")
		print len(time), len(BatchThetaDiff)
		axa[1].plot(time, ThetaDiffList, label = 'ThetaDiff')
		axa[1].plot(time, CoThetaDiffList, label = 'CoThetaDiff')
		axa[1].legend()
		axa[1].set_xlabel("Iteration")
		axa[1].set_ylabel("L2 Diff")


if __name__ == '__main__':
	iterations = 100
	dimension =5
	alpha  = .3 
	lambda_ = 0.3   # Inialize A
	gamma = .4  # parameter inc Exp3
	epsilon = .2  # parameter in Epsilon greedy

	n_articles = 1000
	ArticleGroups = 5

	n_users = 10
	UserGroups = 5
	

	poolSize = 10
	batchSize = 10


	userFilename = os.path.join(sim_files_folder, "users_"+str(n_users)+"+dim-"+str(dimension)+ "Ugroups" + str(UserGroups)+".json")
	resultsFile = os.path.join(result_folder, "Results.csv")

	"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
	UM = UserManager(dimension, n_users, UserGroups = UserGroups, thetaFunc=featureUniform, argv={'l2_limit':1})
	#users = UM.simulateThetafromUsers()
	#UM.saveUsers(users, userFilename, force = False)
	users = UM.loadUsers(userFilename)
	


	articlesFilename = os.path.join(sim_files_folder, "articles_"+str(n_articles)+"+dim"+str(dimension) + "Agroups" + str(ArticleGroups)+".json")
	AM = ArticleManager(dimension, n_articles=n_articles, ArticleGroups = ArticleGroups,
			FeatureFunc=featureUniform,  argv={'l2_limit':1})
	#articles = AM.simulateArticlePool()
	#AM.saveArticles(articles, articlesFilename, force=False)
	articles = AM.loadArticles(articlesFilename)

	simExperiment = simulateOnlineData(dimension  = dimension,
						iterations = iterations,
						articles=articles,
						users = users,		
						noise = lambda : np.random.normal(scale = .0001),
						batchSize = batchSize,
						type_ = "UniformTheta", environmentVars={},
						signature = AM.signature,
						poolarticleSize = poolSize
				)
	print "Starting for ", simExperiment.simulation_signature

	algorithms = {}
	algorithms['LinUCB'] = LinUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	algorithms['CoLinUCB'] =  CoLinUCBAlgorithm(dimension=dimension, alpha=alpha, lambda_ = lambda_, n = n_users, W= simExperiment.getW())
	#algorithms['e-greedy'] = EpsilonGreedy_Multi_Algorithm(epsilon = epsilon, n = n_users)
	simExperiment.runAlgorithms(algorithms)



	
