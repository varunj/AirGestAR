import numpy as np
np.random.seed(123)
import glob, os
import pandas as pd
from scipy.spatial.distance import euclidean
from keras.layers.core import Dropout, Activation, Flatten
import matplotlib.pyplot as plt
from math import sqrt
from pprint import pprint
import pdb
np.set_printoptions(threshold=np.nan)
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import time
import scipy as sp
import scipy.interpolate
import pickle
import pprint

CLASSES = ('bloom','click','zooin','zoout')
NOS_CLASSES = len(CLASSES)


model = load_model('../lstm_models/my_model.h5')
model.load_weights('../lstm_models/my_model_weights.h5')

dataSeq = []
targetSeq = []

dic = pickle.load(open("../result_dics/dic_train_1_2d_100.pickle", "rb" ))
for x,y in dic.items():
	y = y.reshape(100, 42)
	y = np.transpose(y)
	y = y.reshape(1, 42, 100)
	dataSeq.append(y)
	targetarr = np.zeros(NOS_CLASSES)
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==x[:5]):
			targetarr[i] = 1
	targetSeq.append(targetarr)

data = np.vstack(dataSeq)
target = np.vstack(targetSeq)
print(len(dataSeq))
print(data.shape)
print(target.shape)

# {sno: [true, pred, , trueProb, resultProb]}
c = 0
dic = {}
for eachData in data:
	eachData = eachData.reshape(1, 42, 100)
	result = model.predict(eachData)
	dic[c+1] = [CLASSES[np.argmax(target[c])], CLASSES[np.argmax(result)], target[c], result[0]]
	c = c + 1

print(dic)
with open('result.pickle', 'wb') as handle:
	pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
