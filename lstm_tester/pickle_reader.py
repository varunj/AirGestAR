import numpy as np
np.random.seed(123)
import glob, os
import matplotlib.pyplot as plt
from math import sqrt
from pprint import pprint
np.set_printoptions(threshold=np.nan)
import time
import scipy as sp
import scipy.interpolate
import pickle
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score

CLASSES = ('bloom','click','zooin','zoout')
NOS_CLASSES = len(CLASSES)

# {sno: [true, pred, , trueProb, resultProb]}
dic = pickle.load(open("result.pickle", "rb"))

preMicro = []
recMicro = []
f1micro = []
preMacro = []
recMacro = []
f1Macro = []
f1Micro = []
thrsh = []

for acc in range(1,95):
	print('------------------------------ acc = ' + str(acc) + ' ------------------------------')
	acc = acc/100.0
	trueArr = []
	predArr = []
	for x in dic:
		# 1/2/3/4 for classified. 0 for unclassified
		true = np.argmax(dic[x][2])+1
		pred = 0
		if (max(dic[x][3]) >= acc):
			pred = np.argmax(dic[x][3])+1
			trueArr.append(str(true))
			predArr.append(str(pred))


	confMatrix = confusion_matrix(trueArr, predArr)

	# micro
	truePositiveSum = confMatrix[0][0]+confMatrix[1][1]+confMatrix[2][2]+confMatrix[3][3]
	falsePositiveSum = confMatrix[0][1]+confMatrix[0][2]+confMatrix[0][3] + \
						confMatrix[1][0]+confMatrix[1][2]+confMatrix[1][3] + \
						confMatrix[2][0]+confMatrix[2][1]+confMatrix[2][3] + \
						confMatrix[3][0]+confMatrix[3][1]+confMatrix[3][2]
	preMicro.append(truePositiveSum*1.0/(truePositiveSum+falsePositiveSum))
	recMicro.append(truePositiveSum*1.0/80)
	p = preMicro[len(preMicro)-1]
	r = recMicro[len(recMicro)-1]
	f1Micro.append(2*p*r/(p+r))
	
	# macro
	preMacro.append(((confMatrix[0][0]*1.0/(confMatrix[0][0]+confMatrix[1][0]+confMatrix[2][0]+confMatrix[3][0])) + \
		(confMatrix[1][1]*1.0/(confMatrix[0][1]+confMatrix[1][1]+confMatrix[2][1]+confMatrix[3][1])) + \
		(confMatrix[2][2]*1.0/(confMatrix[0][2]+confMatrix[1][2]+confMatrix[2][2]+confMatrix[3][2])) + \
		(confMatrix[3][3]*1.0/(confMatrix[0][3]+confMatrix[1][3]+confMatrix[2][3]+confMatrix[3][3])))/4)
	recMacro.append(((confMatrix[0][0]*1.0/(20)) + \
		(confMatrix[1][1]*1.0/(20)) + \
		(confMatrix[2][2]*1.0/(20)) + \
		(confMatrix[3][3]*1.0/(20)))/4)
	p = preMacro[len(preMacro)-1]
	r = recMacro[len(recMacro)-1]
	f1Macro.append(2*p*r/(p+r))

	print(confusion_matrix(trueArr, predArr))
	print('prmicro: ' + str(truePositiveSum/(truePositiveSum+falsePositiveSum)*1.0))
	print('remicro: ' + str(truePositiveSum/80.0))
	print('accuracy: ' + str((confMatrix[0][0]+confMatrix[1][1]+confMatrix[2][2]+confMatrix[3][3])*1.0/80))
	thrsh.append(acc)

plt.plot(preMicro, recMicro,  marker='+')
plt.title('P/R-R/P Micro')
plt.savefig('graph_pr_micro.png')
plt.close()
plt.plot(preMacro, recMacro, marker='+')
plt.title('P/R-R/P Macro')
plt.savefig('graph_pr_macro.png')
plt.close()
plt.plot(thrsh, f1Macro, 'ro')
plt.title('f1/threshold Macro')
plt.savefig('graph_f1_macro.png')
plt.close()
plt.plot(thrsh, f1Micro, 'ro')
plt.title('f1/threshold Micro')
plt.savefig('graph_f1_micro.png')
plt.close()

print(f1Macro)
print(f1Micro)
print('max f1macro: ' + str(max(f1Macro)) + ' at thresh: ' + str(thrsh[np.argmax(f1Macro)]))
print('max f1micro: ' + str(max(f1Micro)) + ' at thresh: ' + str(thrsh[np.argmax(f1Micro)]))