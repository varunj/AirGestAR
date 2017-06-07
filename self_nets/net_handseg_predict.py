# THEANO_FLAGS=device=gpu,floatX=float32 python net_handseg_predict.py
import theano

import numpy as np
np.set_printoptions(threshold=np.nan)
np.random.seed(123)
import random
random.seed(123)

from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten
from keras.utils import np_utils
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K

import os, glob
import cv2

PATH_COLOR = './data/evaluation/color/'
NOS_INP = 10

def makeChannelsSecDimen(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (256, 256)) 
	a = np.reshape(np.array(img[:,:,0]), (1, 256, 256))
	b = np.reshape(np.array(img[:,:,1]), (1, 256, 256))
	img1 = np.append(a, b, axis=0)
	c = np.reshape(np.array(img[:,:,2]), (1, 256, 256))
	toReturn = np.append(img1, c, axis=0)
	return toReturn/255.0

# make list of all files to be read. then shuffle it
fileNamesArr = []
for fileName in glob.glob(PATH_COLOR + "*.png"):
	fileNamesArr.append(os.path.basename(fileName))
fileNamesArr = random.sample(fileNamesArr, len(fileNamesArr))

# read images to imageStacks. make nos channels second dimen. scale input to 0-1
c = 0
trainInpStack = np.zeros((NOS_INP, 3, 256, 256))

for eachImgName in fileNamesArr[:NOS_INP]:
	trainInpPath = PATH_COLOR + eachImgName
	trainInpStack[c] = makeChannelsSecDimen(trainInpPath).reshape(1, 3, 256, 256)
	c = c+1
	print 'done for image #' + str(c)

modelJsonFile = open('model0.json', 'r')
modelJson = modelJsonFile.read()
modelJsonFile.close()
model = model_from_json(modelJson)

model.load_weights("model0.h5")

# evaluate loaded model on test data
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc', 'mae'])
prediction = model.predict(trainInpStack)

print prediction.shape
c = 0
for x in prediction:
	print x.shape
	print np.amax(x)
	cv2.imwrite(fileNamesArr[c].split('.')[0] + '_out.png', x*255)
	c = c+1