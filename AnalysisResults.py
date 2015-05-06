import numpy as np
import os
from conf import *
from matplotlib.pylab import *
from operator import itemgetter
import random

filenamesResult = [x for x in os.listdir(save_addressResult) if 'csv' in x]
userNum = 10


LinUCBuserReward = {}
LinUCBuserOptimalReward = {}
LinUCBuserRegret = {}

CoLinUCBuserReward = {}
CoLinUCBuserOptimalReward = {}
CoLinUCBuserRegret = {}

Exp3userReward = {}
Exp3userOptimalReward = {}
Exp3userRegret = {}

for i in range(userNum+1):
	LinUCBuserReward[i] = []
	LinUCBuserOptimalReward[i] = []
	LinUCBuserRegret[i] = []
	CoLinUCBuserReward[i] = []
	CoLinUCBuserOptimalReward[i] = []
	CoLinUCBuserRegret[i] = []

for x in filenamesResult:
	filename = os.path.join(save_addressResult, x)
	print x


	if 'Revised_LinUCB_AccReward' in x:
		iter_ = []
		with open(filename, 'r') as f:
			lineNum = -1
			for line in f:
				lineNum += 1
				if lineNum >=1:
					words = line.split(',')
					iter_.append(words[0])
					for i in range(1, userNum +2):
						LinUCBuserReward[i-1].append(float(words[i]))
			print 'LinReward', len(LinUCBuserReward[1])
     
	if 'Revised_LinUCB_AccOptimalReward' in x:
		with open(filename, 'r') as f:
			lineNum = -1
			for line in f:
				lineNum += 1
				if lineNum >=1:
					words = line.split(',')
					for i in range(1, userNum +2):
						LinUCBuserOptimalReward[i-1].append(float(words[i]))
			print 'LinReward', len(LinUCBuserOptimalReward[1])

	if 'Revised_LinUCB_AccRegret' in x:
		with open(filename, 'r') as f:
			lineNum = -1
			for line in f:
				lineNum += 1
				if lineNum >=1:
					words = line.split(',')
					for i in range(1, userNum +2):
						LinUCBuserRegret[i-1].append(float(words[i]))
			print 'LinRegret', len(LinUCBuserRegret[1])

	if 'Revised_CoUCB_AccReward' in x:
		with open(filename, 'r') as f:
			lineNum = -1
			for line in f:
				lineNum += 1
				if lineNum >=1:
					words = line.split(',')
					for i in range(1, userNum +2):
						CoLinUCBuserReward[i-1].append(float(words[i]))
			print 'Co Reward', len(CoLinUCBuserReward[1])
	if 'Revised_CoUCB_AccOptimalReward' in x:
		with open(filename, 'r') as f:
			lineNum = -1
			for line in f:
				lineNum += 1
				if lineNum >=1:
					words = line.split(',')
					for i in range(1, userNum +2):
						CoLinUCBuserOptimalReward[i-1].append(float(words[i]))
			print 'Co OpReward', len(CoLinUCBuserOptimalReward[1])
	if 'Revised_CoUCB_AccRegret' in x:
		with open(filename, 'r') as f:
			lineNum = -1
			for line in f:
				lineNum += 1
				if lineNum >=1:
					words = line.split(',')
					for i in range(1, userNum +2):
						CoLinUCBuserRegret[i-1].append(float(words[i]))
			print 'COLINUCB-regert', len(CoLinUCBuserRegret[1])

print len(iter_), len(LinUCBuserRegret[1])

#Batch
Batch_LinUCBRegret = {}
Batch_CoLinUCbRegret = {}
LinUCBRegretRatio = {}   # Accumulated regret/ Accumulated Optimal Reward
CoLinUCBRegretRatio = {}
CoLinUCBtoLinUCBRegretRatio = {}
TempLinUCBRegret = {}
TempCoLinUCBRegret = {}
batchSize = 5
# the last one is average over users
for x in range(userNum + 1):
	Batch_LinUCBRegret[x] = []
	Batch_CoLinUCbRegret[x] = []
	LinUCBRegretRatio[x] = []
	CoLinUCBRegretRatio[x] =[]
	CoLinUCBtoLinUCBRegretRatio[x] = []
	TempLinUCBRegret[x] = []
	TempCoLinUCBRegret[x] = []

for x in range(userNum + 1):
	BIter_  = []
	batchIter_ = 0
	for i in range(len(LinUCBuserRegret[1])):
		TempLinUCBRegret[x].append(LinUCBuserRegret[x][i])
		TempCoLinUCBRegret[x].append(CoLinUCBuserRegret[x][i])	
		if i%batchSize == 0 and i > 0:
			batchIter_ +=1
			BIter_.append(i)
			Batch_LinUCBRegret[x].append(sum(TempLinUCBRegret[x])/batchSize)
			Batch_CoLinUCbRegret[x].append(sum(TempCoLinUCBRegret[x])/batchSize)

			LinUCBRegretRatio[x].append(sum(LinUCBuserRegret[x][(batchIter_-1)*batchSize : batchIter_*batchSize]) /sum(LinUCBuserOptimalReward[x][(batchIter_-1)*batchSize : batchIter_*batchSize]))
			CoLinUCBRegretRatio[x].append(sum(CoLinUCBuserRegret[x][(batchIter_-1)*batchSize : batchIter_*batchSize]) / sum(CoLinUCBuserOptimalReward[x][(batchIter_-1)*batchSize : batchIter_*batchSize]))
			CoLinUCBtoLinUCBRegretRatio[x].append(sum(CoLinUCBuserRegret[x][(batchIter_-1)*batchSize : batchIter_*batchSize]) / sum(LinUCBuserRegret[x][(batchIter_-1)*batchSize : batchIter_*batchSize]))
			TempLinUCBRegret[x] = []
			TempCoLinUCBRegret[x] = []
print len(BIter_), len(Batch_LinUCBRegret[1])

for i in range(userNum):

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, sharey = False)
	ax1.plot(BIter_, Batch_LinUCBRegret[i], label = 'LinUCB Regret')
	ax1.plot(BIter_, Batch_CoLinUCbRegret[i], label = 'CoLinUCB Regret')

	ax2.plot(BIter_, LinUCBRegretRatio[i], label = 'Lin Reg-Opt Ratio')
	ax2.plot(BIter_, CoLinUCBRegretRatio[i], label = 'CoLin Reg-Opt Ratio')
	ax3.plot(BIter_, CoLinUCBtoLinUCBRegretRatio[i], label = 'CoLin-Lin Regret Ratio')
	ax1.set_title('user' + str(i))
	ax1.legend(loc = 'lower right')
	ax2.legend(loc = 'upper right')
	ax3.legend(loc = 'lower right')
'''
f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, sharey = False)
ax1.plot(BIter_, Batch_LinUCBRegret[10], label = 'LinUCB Regret')
ax1.plot(BIter_, Batch_CoLinUCbRegret[10], label = 'CoLinUCB Regret')

ax2.plot(BIter_, LinUCBRegretRatio[10], label = 'Lin Reg-Opt Ratio')
ax2.plot(BIter_, CoLinUCBRegretRatio[10], label = 'CoLin Reg-Opt Ratio')
ax3.plot(BIter_, CoLinUCBtoLinUCBRegretRatio[10], label = 'CoLin-Lin Regret Ratio')
ax1.set_title('Average')
ax1.legend(loc = 'lower right')
ax2.legend(loc = 'upper right')
ax3.legend(loc = 'lower right')

'''
