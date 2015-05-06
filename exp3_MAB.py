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
    def __init__(self, gamma, id):
        self.id = id
        self.gamma = gamma
        self.weights = 1.0
        self.pta = 0.0
    def updateParameters(self, n_arms, reward):
        self.updateWeight(n_arms, reward)
        self.updatePta(n_arms, total_Weights)

    def updateWeight(self, n_arms, reward):
        X = reward / self.pta
        growth_factor = math.exp((self.gamma / n_arms)*X)
        self.weights = self.weights * growth_factor
    def updatePta(self, n_arms, total_weight):
        self.pta = (1-self.gamma) * (self.weights / total_weight)
        self.pta +=(self.gamma)*(1.0/float(n_arms))
      
class UCB1Struct:
    def __init__(self, id):
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
    def __init__(self, id):
        self.id = id
        self.totalReward = 0.0
        self.numPlayed = 0
        self.pta = 0.0
    def reInitilize(self):
        self.totalReward = 0.0
        self.numPlayed = 0
        self.pta = 0.0
    def updateParameter(self, click):
        self.totalReward += click
        self.numPlayed += 1
    def updatePta(self):
        try:
            self.pta = self.totalReward / self.numPlayed
        except ZeroDivisionError:
            self.pta = 0.0
        #return self.pta
        
        
class Exp3Algorithm:
    def __init__(self, dimension, gamma, decay = None):
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
        total_Weights = 0.0
        for x in pool_articles:
            if x.id not in self.articles:
                self.articles[x.id] = Exp3Struct(self.gamma, x.id)
            total_Weights += self.articles[x.id].weights
        for x in pool_articles:
            self.articles[x.id].updatePta(len(pool_articles), total_Weights)
            cum_pta += self.articles[x.id].pta
            if cum_pta >r:
                return x
        return choice(pool_articles)
    # parameters : (pickedArticle, Nun of articles in article pool, click)
    def updateParameters(self, pickedArticle, click, userID): 
        self.articles[pickedArticle.id].updateWeight(self.PoolArticleNum, click)

class UCB1Algorithm:
    def __init__(self, dimension):
        self.articles = {}
        self.dimension = dimension
    def decide(self, pool_articles, userID, time_): #parameters:(article pool, number of times that has been played)
        articlePicked = None
        for x in pool_articles:
            if x.id not in self.articles:
                self.articles[x.id] = UCB1Struct(x.id)
        
        allNumPlayed = sum([self.articles[x.id].numPlayed for x in pool_articles])

        for x in pool_articles:
            self.articles[x.id].updatePta(allNumPlayed)
            
            if self.articles[x.id].numPlayed == 0:
                articlePicked = x
                return articlePicked
        return max(np.random.permutation([(x, self.articles[x.id].pta) for x in pool_articles]), key = itemgetter(1))[0]

            
    def updateParameters(self, pickedArticle, click, userID ):  #parameters: (pickedArticle, click)
        self.articles[pickedArticle.id].updateParameter( click)

class EpsilonGreedyAlgorithm:
    def __init__(self, dimension, epsilon, decay = None):
        self.articles = {}
        self.decay = decay
        self.dimension = dimension
        self.epsilon = epsilon
    def decide(self, pool_articles, userID, time_):
        article_Picked = None
        #if random.random() < self.epsilon:
        #   article_Picked = choice(pool_articles)
        #else:
        for x in pool_articles:
            if x.id not in self.articles:
                self.articles[x.id] = GreedyStruct(x.id)
            self.articles[x.id].updatePta()
        if random.random() < self.epsilon:
            article_Picked = choice(pool_articles)
        else:
            article_Picked = max(np.random.permutation([(x, self.articles[x.id].pta) for x in pool_articles]), key = itemgetter(1))[0]
        return article_Picked
    def updateParameters(self, pickedArticle, click, userID):
        self.articles[pickedArticle.id].updateParameter(click)    
    
    def getarticleCTR(self, article_id):
        return self.articles[article_id].stats.CTR
    
    def getLearntParams(self, article_id):
        return np.zeros(self.dimension)

    
        
        

                
            
                    
                
                
        
