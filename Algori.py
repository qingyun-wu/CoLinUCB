import numpy as np

class LinUCBUserStruct(object):
	def __init__(self, featureDimension, userID, lambda_):
		self.userID = userID
		self.A = lambda_*np.identity(n = featureDimension)
		self.b = np.zeros(featureDimension)
		self.UserTheta = np.zeros(featureDimension)
		self.lambda_ = lambda_

	def PreUpdateParameters(self):
		pass

	def updateParameters(self, articlePicked, click):
		featureVector = articlePicked.featureVector
		self.A += np.outer(featureVector, featureVector)
		self.b += featureVector*click
		self.UserTheta = np.dot(np.linalg.inv(self.A), self.b)

	def getProb(self, alpha, users, article):
		featureVector = article.featureVector
		mean = np.dot(users[self.userID].UserTheta, featureVector)
		var = np.sqrt(np.dot(np.dot(featureVector, np.linalg.inv(users[self.userID].A)), featureVector))
		pta = mean + alpha * var
		return pta


class CoLinUCBUserStruct(LinUCBUserStruct):
	def __init__(self, featureDimension, userID, lambda_):
		LinUCBUserStruct.__init__(self, featureDimension, userID, lambda_)

		self.LambdaIdentity = lambda_*np.identity(n = featureDimension)
		self.A = np.zeros(shape = (featureDimension, featureDimension))
		self.CoA = np.zeros(shape = (featureDimension, featureDimension))

	def PreUpdateParameters(self, users, W):
		U_id = self.userID
		self.CoA = sum([(W[m][U_id]**2) * users[m].A for m in range(W.shape[0])])

		Cob = np.zeros(self.b.shape[0])
		for m in range(W.shape[0]):
			NeighborTheta = sum([W[m][j] * users[j].UserTheta for j in range(W.shape[1])]) - W[m][U_id]*users[U_id].UserTheta
			Cob += W[m][U_id] * (users[m].b - np.dot(users[m].A, NeighborTheta))
		
		self.UserTheta = np.dot(np.linalg.inv(self.LambdaIdentity + self.CoA), Cob)

	def updateParameters(self, articlePicked, click, users, W):
		featureVector = articlePicked.featureVector		

		self.A += np.outer(featureVector, featureVector)
		self.b += featureVector*click

		self.PreUpdateParameters(users, W)

	def getProb(self, alpha, users, article, W):
		U_id = self.userID

		#Compute Collaborative-theta
		CoTheta = sum([W[U_id][j]*users[j].UserTheta for j in range(W.shape[1])])
		#Compute weighted sum of CoA
		CCA = self.LambdaIdentity + sum([W[U_id][j]*users[j].CoA for j in range(W.shape[1])])
			
		self.mean = np.dot(CoTheta, article.featureVector)
		self.var = np.sqrt(np.dot(np.dot(article.featureVector, np.linalg.inv(CCA)), article.featureVector))
		self.pta = self.mean + alpha * self.var
		return self.pta	


class LinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, decay = None):  # n is number of users
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(LinUCBUserStruct(dimension, i, lambda_ )) 

		self.dimension = dimension
		self.alpha = alpha

	def decide(self, pool_articles, userID):

		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, self.users, x)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked
	def PreUpdateParameters(self, userID):
		self.users[userID].PreUpdateParameters()

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click)
		
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
		maxPTA = float('-inf')
		articlePicked = None
		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, self.users,  x, self.W)
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked

	def PreUpdateParameters(self, userID):
		self.users[userID].PreUpdateParameters(self.users, self.W)

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click, self.users, self.W)

	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta
		
	def getCoThetaFromCoLinUCB(self, userID):
		return sum([self.W[userID][j]*self.users[j].UserTheta for j in range(self.W.shape[1])])



