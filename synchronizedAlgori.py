import numpy as np

def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W


class LinUCBUserStruct:
	def __init__(self, featureDimension, userID, lambda_):
		self.userID = userID
		self.A = lambda_*np.identity(n = featureDimension)
		self.b = np.zeros(featureDimension)
		self.UserTheta = np.zeros(featureDimension)

	def updateParameters(self, articlePicked, click):
		featureVector = articlePicked.featureVector
		self.A += np.outer(featureVector, featureVector)
		self.b += featureVector*click
		self.UserTheta = np.dot(np.linalg.inv(self.A), self.b)
		
	def getTheta(self):
		return self.UserTheta
	
	def getA(self):
		return self.A

	def getProb(self, alpha, users, article):
		featureVector = article.featureVector
		mean = np.dot(self.getTheta(), featureVector)
		var = np.sqrt(np.dot(np.dot(featureVector, np.linalg.inv(self.getA())), featureVector))
		pta = mean + alpha * var
		return pta


def CoLinUCBgetProb(alpha, article, userID, theta, CCA, userNum ):
		featureVectorM = np.zeros(shape =(len(article.featureVector), userNum ))
		featureVectorM.T[userID] = article.featureVector
		featureVectorV = vectorize(featureVectorM)

		mean = np.dot(theta, article.featureVector)
		var = np.sqrt(np.dot(np.dot(featureVectorV, CCA), featureVectorV))
		pta = mean + alpha * var
		return pta

class CoLinUCBUserSharedStruct:
	def __init__(self, featureDimension, lambda_, userNum):
		self.userNum = userNum
		self.A = lambda_*np.identity(n = featureDimension* userNum)
		self.CCA = np.identity(n = featureDimension* userNum)
		self.b = np.zeros(featureDimension*userNum)

		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		self.CoTheta = np.zeros(shape = (featureDimension, userNum))

		self.featureVectorMatrix = np.zeros(shape =(featureDimension, userNum) )
		self.reward = np.zeros(userNum)

	def updateParameters(self, articlePicked, click,  W, userID):	
		self.featureVectorMatrix.T[userID] = articlePicked.featureVector
		self.reward[userID] = click
		featureDimension = len(self.featureVectorMatrix.T[userID])

		current_A = np.zeros(shape = (featureDimension* self.userNum, featureDimension*self.userNum))
		current_b = np.zeros(featureDimension*self.userNum)		
		for i in range(self.userNum):
			X = vectorize(np.outer(self.featureVectorMatrix.T[i], np.transpose(W.T[i]))) 
			XS =  np.outer(X, X)	
			current_A +=XS
			current_b += self.reward[i] * X
	
		self.A += current_A
		self.b += current_b

		self.UserTheta =  matrixize(np.dot(np.linalg.inv(self.A), self.b), featureDimension ) 
		self.CoTheta = np.dot(self.UserTheta, W)

		#self.CCA = np.dot( np.kron(W, np.identity(n=featureDimension)) , np.linalg.inv(self.A))
		BigW = np.kron(W, np.identity(n=featureDimension))
		self.CCA = np.dot(np.dot(BigW , np.linalg.inv(self.A)), np.transpose(BigW))

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
		pass
	
	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click)
		
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta

class CoLinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, W):  # n is number of users
		self.USERS = CoLinUCBUserSharedStruct(dimension, lambda_, n)
		self.dimension = dimension
		self.alpha = alpha
		self.W = W

	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = CoLinUCBgetProb(self.alpha, x, userID, self.USERS.CoTheta.T[userID], self.USERS.CCA, self.USERS.userNum)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked
	def PreUpdateParameters(self, userID):
		pass
	def updateParameters(self, articlePicked, click, userID):
		self.USERS.updateParameters(articlePicked, click,  self.W, userID)
		
	def getLearntParameters(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoThetaFromCoLinUCB(self, userID):
		return self.USERS.CoTheta.T[userID]

	def getA(self):
		return np.linalg.inv(self.USERS.A) 



