import numpy as np
from operator import itemgetter      #for easiness in sorting and finding max and stuff
from matplotlib.pylab import *
from random import sample
from scipy.sparse import csgraph 
import os
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, result_folder, save_address
from util_functions import *
from Articles import *
from Users import *
from Algori import *

class simulateOnlineData():
	def __init__(self, dimension, iterations, articles, users, 
					batchSize = 1000,
					noise = lambda : 0,
					type_ = 'UniformTheta', 
					signature = '', 
					poolArticleSize = 10, 
					NoiseScale = 0):

		self.simulation_signature = signature
		self.type = type_

		self.dimension = dimension
		self.iterations = iterations
		self.noise = noise
		self.articles = articles 
		self.users = users

		self.poolArticleSize = poolArticleSize
		self.batchSize = batchSize
		
		self.W = self.initializeW()
		self.NoiseScale = NoiseScale
	
	# create user connectivity graph
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
		W = I - epsilon * L  # W is a double stochastic matrix
		return W

	def getW(self):
		return self.W

	def batchRecord(self, iter_):
		print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime

	def regulateArticlePool(self):
		self.articlePool = sample(self.articles, self.poolArticleSize)

	def CoTheta(self):
		for ui in self.users:
			ui.CoTheta = np.zeros(self.dimension)
			for uj in self.users:
				ui.CoTheta += self.W[uj.id][ui.id] * np.asarray(uj.theta)
			print 'Users', ui.id, 'CoTheta', ui.CoTheta

	def getReward(self, user, pickedArticle):
		return np.dot(user.CoTheta, pickedArticle.featureVector)

	def GetOptimalReward(self, user, articlePool):		
		maxReward = sys.float_info.min
		for x in articlePool:	 
			reward = self.getReward(user, x)
			if reward > maxReward:
				maxReward = reward
		return maxReward
	
	def getThetaDiff(self, user, theta):
		return np.linalg.norm(user.theta - theta)

	def getCoThetaDiff(self, user, cotheta):	
		return np.linalg.norm(user.CoTheta - cotheta)

	def runAlgorithms(self, algorithms):
		preUpdateFlag = True

		# get cotheta for each user
		self.CoTheta()
		self.startTime = datetime.datetime.now()

		tim_ = []
		BatchAverageRegret = {}
		AccRegret = {}
		
		# Initialization
		for alg_name in algorithms.iterkeys():
			BatchAverageRegret[alg_name] = []
			AccRegret[alg_name] = {}

			for i in range(len(self.users)):
				AccRegret[alg_name][i] = []

		CoLinUCB_ThetaDiffList = []
		CoLinUCB_CoThetaDiffList = []
		LinUCB_CoThetaDiffList = []

		# Loop begin
		for iter_ in range(self.iterations):
			CoLinUCB_ThetaDiff = 0
			CoLinUCB_CoThetaDiff = 0
			LinUCB_CoThetaDiff = 0
			for u in self.users:
				self.regulateArticlePool() # select random articles

				noise = self.noise()
				#get optimal reward for user x at time t
				OptimalReward = self.GetOptimalReward(u, self.articlePool) + noise

				for alg_name, alg in algorithms.items():
					if preUpdateFlag == True:
						alg.PreUpdateParameters(u.id)

					pickedArticle = alg.decide(self.articlePool, u.id)
					reward = self.getReward(u, pickedArticle) + noise
					alg.updateParameters(pickedArticle, reward, u.id)

					regret = OptimalReward - reward	
					AccRegret[alg_name][u.id].append(regret)
					
					if alg_name == 'CoLinUCB':
						CoLinUCB_ThetaDiff += self.getThetaDiff(u, alg.getLearntParameters(u.id))
						CoLinUCB_CoThetaDiff += self.getCoThetaDiff(u, alg.getCoThetaFromCoLinUCB(u.id))

					elif alg_name == 'LinUCB':
						LinUCB_CoThetaDiff += self.getCoThetaDiff(u, alg.getLearntParameters(u.id))

			# how do we know we will have those two algorithms??
			CoLinUCB_ThetaDiffList.append(CoLinUCB_ThetaDiff/len(self.users))
			CoLinUCB_CoThetaDiffList.append(CoLinUCB_CoThetaDiff/len(self.users))
			LinUCB_CoThetaDiffList.append(LinUCB_CoThetaDiff/len(self.users))

			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name in algorithms.iterkeys():
					TotalAccRegret = sum(sum (u) for u in AccRegret[alg_name].itervalues())
					BatchAverageRegret[alg_name].append(TotalAccRegret)
		
		# plot the results		
		f, axa = plt.subplots(2, sharex=True)

		for alg_name in algorithms.iterkeys():		
			axa[0].plot(tim_, BatchAverageRegret[alg_name], label = alg_name)
			axa[0].lines[-1].set_linewidth(1.5)
			axa[0].legend()
			axa[0].set_xlabel("Iteration")
			axa[0].set_ylabel("Regret")
			axa[0].set_title("Noise scale = " + str(self.NoiseScale) + ' Pre=' + str(preUpdateFlag))
			axa[0].lines[-1].set_linewidth(1.5)
		
		time = range(self.iterations)
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
		
		plt.show()


if __name__ == '__main__':
	iterations = 1000
	NoiseScale = 0.001
	dimension = 5
	alpha  = 0.3 
	lambda_ = 0.2   # Inialize A

	n_articles = 1000
	ArticleGroups = 5

	n_users = 10
	UserGroups = 5	

	poolSize = 20
	batchSize = 10
	
	userFilename = os.path.join(sim_files_folder, "users_"+str(n_users)+"+dim-"+str(dimension)+ "Ugroups" + str(UserGroups)+".json")
	
	#"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
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
						poolArticleSize = poolSize, NoiseScale = NoiseScale)
	print "Starting for ", simExperiment.simulation_signature

	algorithms = {}
	algorithms['LinUCB'] = LinUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	algorithms['CoLinUCB'] = CoLinUCBAlgorithm(dimension=dimension, alpha= alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW())
	
	simExperiment.runAlgorithms(algorithms)



	
