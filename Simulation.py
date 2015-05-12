import math
import numpy as np
from matplotlib.pylab import *
from random import random, sample, randint
from scipy.stats import lognorm
import json
import os
import sys
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize
# local address to save simulated users, simulated articles, and results
from conf_Qingyun import sim_files_folder, result_folder, save_address
from util_functions import *
from Articles import *
from Users import *
from Algori import *

class simulateOnlineData():
	def __init__(self, dimension, iterations, articles, users, 
		batchSize = 1000,
		noise = lambda : 0,
		type_ = 'UniformTheta', 
		signature = '', poolarticleSize = 10, NoiseScale = 0):

		self.simulation_signature = signature
		self.type = type_

		self.dimension = dimension
		self.iterations = iterations
		self.noise = noise
		self.articles = articles 
		self.users = users

		self.poolarticleSize = poolarticleSize
		self.batchSize = batchSize
		
		self.W = self.initializeW()
		self.NoiseScale = NoiseScale
	

	def initializeW(self):
		n = len(self.users)
	
		a = np.ones(n-1) 
		b =np.ones(n);
		c = np.ones(n-1)
		k1 = -1
		k2 = 0
		k3 = 1
		A = np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
		G = A
		
		L = csgraph.laplacian(G, normed = False)
		epsilon = 0.2
		I = np.identity(n)
		W = I - epsilon * L  # W is a double stostastic matrix
		return W

	def getW(self):
		return self.W

	def batchRecord(self, iter_):
		print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime

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
	def getCoLinUCB_ThetaAbsDiff(self, user, alg):
		return np.linalg.norm(user.theta - alg.getLearntParameters(user.id))
	def getCoLinUCB_CoThetaAbsDiff(self, user, alg):
		return np.linalg.norm(user.CoTheta - alg.getCoThetaFromCoLinUCB(user.id))

	def getLinUCB_CoThetaAbsDiff(self, user, alg):	
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
			tim_[alg_name] = []
			UserAverageRegret[alg_name] = []
			BatchAverageRegret[alg_name] = []
			CurrentAccRegret[alg_name] = {}
			AccRegret[alg_name] = {}
			regret[alg_name] = {}
			

			for i in range(len(self.users)):
				AccRegret[alg_name][i] = []
				CurrentAccRegret[alg_name][i] = 0.0

		CoLinUCB_ThetaDiffList = []
		CoLinUCB_CoThetaDiffList = []
		LinUCB_CoThetaDiffList = []

		time = []
		# Loop begin
		for iter_ in range(self.iterations):
			CoLinUCB_ThetaDiff = 0
			CoLinUCB_CoThetaDiff = 0
			LinUCB_CoThetaDiff = 0
			time.append(iter_)
			for x in  range(len(self.users)):

				self.regulateArticlePool()

				noise = self.noise()
				#get optimal Article for user x at time t
				optimalArticle = self.GetOptimalArticle(self.users[x], self.articlePool)
				OptimalReward = self.getReward(self.users[x], optimalArticle) + noise

				for alg_name, alg in algorithms.items():
					pickedArticle[alg_name] = alg.decide(self.articlePool, self.users[x].id)
					reward[alg_name] = self.getReward(self.users[x], pickedArticle[alg_name]) + noise
					alg.updateParameters(pickedArticle[alg_name], reward[alg_name], self.users[x].id)

					regret[alg_name][x] = OptimalReward - reward[alg_name]	
					CurrentAccRegret[alg_name][x] = CurrentAccRegret[alg_name][x]+regret[alg_name][x]	
					AccRegret[alg_name][x].append(CurrentAccRegret[alg_name][x])


				CoLinUCB_ThetaDiff += self.getCoLinUCB_ThetaAbsDiff(self.users[x], algorithms['CoLinUCB'])
				CoLinUCB_CoThetaDiff += self.getCoLinUCB_CoThetaAbsDiff(self.users[x], algorithms['CoLinUCB'])
				LinUCB_CoThetaDiff += self.getLinUCB_CoThetaAbsDiff(self.users[x], algorithms['LinUCB'])

			CoLinUCB_ThetaDiffList.append(CoLinUCB_ThetaDiff/(1.0*len(self.users)))
			CoLinUCB_CoThetaDiffList.append(CoLinUCB_CoThetaDiff/(1.0*len(self.users)))
			LinUCB_CoThetaDiffList.append(LinUCB_CoThetaDiff/(1.0*len(self.users)))

			for alg_name, alg in algorithms.items():
				UserAverageRegret[alg_name].append((sum([AccRegret[alg_name][i][-1] for i in range(len(users))]) / len(users)))
				if (iter_+1)%self.batchSize ==0 and iter_ >=0:
					self.batchRecord(iter_)
					tim_[alg_name].append(iter_)
					BatchAverageRegret[alg_name].append(sum(UserAverageRegret[alg_name]) / (1.0* self.batchSize))
					UserAverageRegret[alg_name] = []
					
		f, axa = plt.subplots(2, sharex=True)

		for alg_name, alg in algorithms.items():		
			axa[0].plot(tim_[alg_name], BatchAverageRegret[alg_name], label = alg_name)
			axa[0].lines[-1].set_linewidth(1.5)
			axa[0].legend()
			axa[0].set_xlabel("Iteration")
			axa[0].set_ylabel("Regret")
			axa[0].set_title("Noise scale = " + str(self.NoiseScale))
			axa[0].lines[-1].set_linewidth(1.5)
		
   		axa[1].plot(time, CoLinUCB_CoThetaDiffList, label = 'CoLinUCB_CoTheta')
		axa[1].lines[-1].set_linewidth(1.5)
		axa[1].plot(time, LinUCB_CoThetaDiffList, label = 'LinUCB_CoTheta')
		axa[1].lines[-1].set_linewidth(1.5)
		axa[1].plot(time, CoLinUCB_ThetaDiffList, label = 'CoLinUCB_Theta')
		axa[1].lines[-1].set_linewidth(1.5)	
		
		axa[1].legend()
		axa[1].set_xlabel("Iteration")
		axa[1].set_ylabel("SqRoot L2 Diff")
		axa[1].set_yscale('log')
	


if __name__ == '__main__':
	iterations = 1000
	NoiseScale = 1
	dimension =5
	alpha  = .3 
	lambda_ = 0.3   # Inialize A

	n_articles = 1000
	ArticleGroups = 5

	n_users = 10
	UserGroups = 5	

	poolSize = 10
	batchSize = 10
	
	userFilename = os.path.join(sim_files_folder, "users_"+str(n_users)+"+dim-"+str(dimension)+ "Ugroups" + str(UserGroups)+".json")
	
	"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
	# we can choose to simulate users every time we run the program or simulate users once, save it to 'sim_files_folder', and keep using it.
	UM = UserManager(dimension, n_users, UserGroups = UserGroups, thetaFunc=featureUniform, argv={'l2_limit':1})
	#users = UM.simulateThetafromUsers()
	#UM.saveUsers(users, userFilename, force = False)
	users = UM.loadUsers(userFilename)
	


	articlesFilename = os.path.join(sim_files_folder, "articles_"+str(n_articles)+"+dim"+str(dimension) + "Agroups" + str(ArticleGroups)+".json")
	# Similarly, we can choose to simulate articles every time we run the program or simulate articles once, save it to 'sim_files_folder', and keep using it.
	AM = ArticleManager(dimension, n_articles=n_articles, ArticleGroups = ArticleGroups,
			FeatureFunc=featureUniform,  argv={'l2_limit':1})
	#articles = AM.simulateArticlePool()
	#AM.saveArticles(articles, articlesFilename, force=False)
	articles = AM.loadArticles(articlesFilename)

	simExperiment = simulateOnlineData(dimension  = dimension,
						iterations = iterations,
						articles=articles,
						users = users,		
						noise = lambda : np.random.normal(scale = NoiseScale),
						batchSize = batchSize,
						type_ = "UniformTheta", 
						signature = AM.signature,
						poolarticleSize = poolSize, NoiseScale = NoiseScale
				)
	print "Starting for ", simExperiment.simulation_signature

	algorithms = {}
	algorithms['LinUCB'] = LinUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	algorithms['CoLinUCB'] =  CoLinUCBAlgorithm(dimension=dimension, alpha= alpha, lambda_ = lambda_, n = n_users, W= simExperiment.getW())
	
	simExperiment.runAlgorithms(algorithms)



	
