import numpy as np
np.random.seed(123)
import glob, os
from scipy.spatial.distance import euclidean
from keras.layers.core import Dropout, Activation, Flatten
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt
from pprint import pprint
import pdb
np.set_printoptions(threshold=np.nan)
from keras.models import Sequential
from keras import optimizers	
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
import pickle

CLASSES = ('bloom','click')
# CLASSES = ('zooin','zoout')
# CLASSES = ('bloom','click','zooin','zoout')
NOS_CLASSES = len(CLASSES)

def shuffle_data(labels, seq):
	temp = 0
	for k in seq:
		if (temp ==0):
			labels2 = labels[k:k+1,:]
			temp =temp+1
		else:
			labels2 = np.vstack((labels2, labels[k:k+1,:]))
			temp = temp+1	
	return labels2

dataSeq = []
targetSeq = []

dic = pickle.load(open("./result_dics/dic_bloomclick_2d_100.pickle", "rb" ))
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
dic = pickle.load(open("./result_dics/dic_bloomclick_2_2d_100.pickle", "rb" ))
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

# dic = pickle.load(open("./result_dics/dic_zoom_2d_100.pickle", "rb" ))
# for x,y in dic.items():
# 	y = y.reshape(100, 42)
# 	y = np.transpose(y)
# 	y = y.reshape(1, 42, 100)
# 	dataSeq.append(y)
# 	targetarr = np.zeros(NOS_CLASSES)
# 	for i in range(0, NOS_CLASSES):
# 		if (CLASSES[i]==x[:5]):
# 			targetarr[i] = 1
# 	targetSeq.append(targetarr)
# dic = pickle.load(open("./result_dics/dic_zoom_2_2d_100.pickle", "rb" ))
# for x,y in dic.items():
# 	y = y.reshape(100, 42)
# 	y = np.transpose(y)
# 	y = y.reshape(1, 42, 100)
# 	dataSeq.append(y)
# 	targetarr = np.zeros(NOS_CLASSES)
# 	for i in range(0, NOS_CLASSES):
# 		if (CLASSES[i]==x[:5]):
# 			targetarr[i] = 1
# 	targetSeq.append(targetarr)

print(len(dataSeq))
seq = np.arange(len(dataSeq))
np.random.shuffle(seq)
data = np.vstack(dataSeq)
data = shuffle_data(data, seq)
target = np.vstack(targetSeq)
target = shuffle_data(target, seq)

print(data.shape)
print(target.shape)

model = Sequential()  
model.add(LSTM(200, input_shape=(42,100), return_sequences=False))
# model.add(Flatten())
model.add(Dense(NOS_CLASSES, activation='softmax'))

# sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
history = model.fit(data, target, epochs=300, batch_size=5, verbose=2, validation_split=0.30)

# model.save('my_model.h5')
# model.save_weights('my_model_weights.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')







'''
zooin + zoout
acc 100/96 in ~300 epochs
history = model.fit(data, target, epochs=3000, batch_size=5, verbose=2, validation_split=0.30)

'''