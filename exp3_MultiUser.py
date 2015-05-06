#from conf import *
import time
import re
from random import random, choice
from operator import itemgetter
import datetime
import numpy as np
import math
import random
import Queue



class Exp3Struct:
    def __init__(self, gamma, id, userID):
        self.userID = userID
        self.articles = {}
        self.articlePicked = None

        self.id = id
        self.gamma = gamma
        self.weights = 1.0
        self.pta = 0.0
        self.stats = Stats()
           
    def updateWeight(self, n_arms, reward):
        X = reward / self.pta
        growth_factor = math.exp((self.gamma / n_arms)*X)

        self.weights = self.weights * growth_factor
    def updatePta(self, n_arms, total_weight):
        self.pta = (1-self.gamma) * (self.weights / total_weight)
        self.pta +=(self.gamma)*(1.0/float(n_arms))
        
class UCB1Struct:
    def __init__(self, id, userID):
        self.userID = userID
        self.id = id
        self.totalReward = 0.0
        self.numPlayed = 0
        self.pta = 0.0
        self.stats = Stats()
      
    def updateParameter(self, click):
        self.totalReward += click
        self.numPlayed +=1
        
    def updatePta(self, allNumPlayed):
        try:
            self.pta = self.totalReward / self.numPlayed + np.sqrt(2*np.log(allNumPlayed) / self.numPlayed)
        except ZeroDivisionError:
            self.pta = 0.0
        #return self.pta
    
class GreedyStruct:
    def __init__(self, userID):
        self.userID = userID
        #self.totalReward = {}
        #self.numPlayed = {}
        self.KArticles = {}
        self.pta = 0.0
    '''
    def updateParameter(self, articlePickedID, click):
        self.KArticles[articlePickedID].totalReward += click
        self.KArticles[articlePickedID].numPlayed += 1 
    def getProb(self, articleID):
        try:
            self.pta = self.KArticles[articleID].totalReward / self.KArticles[articleID].numPlayed
        except ZeroDivisionError:
            self.pta = 0.0
        return self.pta
    '''

class GreedyArticles:
    def __init__(self, id):
        self.id = id
        self.totalReward = 0.0
        self.numPlayed = 0.0
    def updateParameter(self, click):
        self.totalReward += click
        self.numPlayed +=1
    def getProb(self, articleID):
        try:
            pta = self.totalReward/self.numPlayed
        except ZeroDivisionError:
            pta = 0.0
        return pta
        
        
class Exp3_Multi_Algorithm:
    def __init__(self, dimension, gamma, n, decay = None):
        self.users = {}
        for i in range(n):
            self.users[i] = UserStruct(i)

        self.articles = {}
        self.gamma = gamma

        self.decay = decay
        self.dimension = dimension
        self.PoolArticleNum = 0
    
    def decide(self, pool_articles, userID, time_): #(paramters: article pool)
        "Should self.PoolArticleNum be total articles or total pool_articles?? Please correct the following line if its wrong."
        self.PoolArticleNum = len(pool_articles)
        r = random.random()
        cum_pta = 0.0        
        total_weight = 0.0
        for x in pool_articles:
            if x.id not in self.users[userID].KArticles:
                self.users[userID].KArticles[x.id] = Exp3Struct(self.gamma, x.id, userID)
            total_weight += self.users[userID].KArticles[x.id].weights
        for x in pool_articles:
            self.users[userID].KArticles[x.id].updatePta(len(pool_articles), total_weight)
            cum_pta +=self.users[userID].KArticles[x.id].pta
            if cum_pta>r:
                return x
        return choice(pool_articles)

    # parameters : (pickedArticle, Nun of articles in article pool, click)
    def updateParameters(self, pickedArticle, click, userID): 
        self.users[userID].KArticles[pickedArticle.id].updateWeight(self.PoolArticleNum, click)
        if self.decay:
            self.applyDecayToAll(1)
    

class UCB1_Multi_Algorithm:
    def __init__(self, dimension, n, decay = None):
        self.users = {}
        for i in range(n):
            self.users[i] = UserStruct(i)
        self.articles = {}
        self.decay = decay
        self.dimension = dimension
    def decide(self, pool_articles, userID, time_): #parameters:(article pool, number of times that has been played)
        articlePicked = None
        for x in pool_articles:
            if x.id not in self.users[userID].KArticles:
                self.users[userID].KArticles[x.id]=UCB1Struct(x.id, userID)
        
        allNumPlayed = sum([self.users[userID].KArticles[x.id].numPlayed for x in pool_articles])

        for x in pool_articles:
            self.users[userID].KArticles[x.id].updatePta(allNumPlayed)
            
            if self.users[userID].KArticles[x.id].numPlayed == 0:
                articlePicked = x
                return articlePicked
        return max(np.random.permutation([(x, self.users[userID].KArticles[x.id].pta) for x in pool_articles]), key = itemgetter(1))[0]

            
    def updateParameters(self, pickedArticle, click, userID ):  #parameters: (pickedArticle, click)
        self.users[userID].KArticles[pickedArticle.id].updateParameter( click)
        if self.decay:
            self.applyDecayToAll(1)

class EpsilonGreedy_Multi_Algorithm:
    def __init__(self, epsilon, n):
        self.users = {}
        for i in range(n):
            self.users[i] = GreedyStruct(i)
        self.epsilon = epsilon

    def decide(self, pool_articles, userID):
        article_Picked = None
        if random.random() < self.epsilon:
            article_Picked = choice(pool_articles)
        else:
            maxPTA = float('-inf')
            for x in pool_articles:
                if x not in self.users[userID].KArticles:
                    self.users[userID].KArticles[x.id] = GreedyArticles(x.id)
                x_pta = self.users[userID].KArticles[x.id].getProb(x.id)

                if maxPTA < x_pta:
                    article_Picked = x
                    maxPTA = x_pta
        return article_Picked
    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].KArticles[articlePicked.id].updateParameter( click) 
        
    def getLearntParameters(self, userID):
        return self.users[userID].UserTheta      
