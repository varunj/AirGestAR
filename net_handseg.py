import numpy as np
np.random.seed(123)
import random
random.seed(123)

from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import theano

import time
time1 = time.time()
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from pylab import figure, axes, pie, title, show
import scipy.misc
import cv2


'''
Reference Material
https://keras.io/layers/convolutional/
https://arxiv.org/pdf/1705.01389.pdf
https://github.com/lmb-freiburg
It was trained for hand segmentation on R-train with a batch size of 8 and using ADAM solver [8]. The network was 
initialized using weights of Wei et al. [22] for layers 1 to 16 and then trained for 40000 iterations using a standard
softmax cross-entropy loss. The learning rate was 1 · 10−5 for the first 20000 iterations, 1 · 10−6 for following 10000
iterations and 1 · 10−7 until the end. Except for random color hue augmentation of 0.1 no data augmentation was
used. From the 320×320 pixel images of the training set a 256×256 crop was taken randomly.

one epoch 			= one forward + backward pass of all the training examples
batch size 			= the number of training examples in one pass
nos of iterations 	= number of passes, each pass using [batch size] number of examples. (one pass = forward+backward) 

nos train = 33k (41258* 8:10)
nos iter = 40k
batch size = 8
hence, epochs = 40k/(33k/8) = 10
'''

PATH_COLOR = './data/training/color/'
PATH_HANDMASK = './data/training/mask_hands/'
IMAGE_ROWS = 256
IMAGE_COLS = 256

def ArgMaxLayer(inp):
	return K.argmax(inp, axis=1)

def makeChannelsSecDimen(path):
	img = cv2.imread(path)
	a = np.reshape(np.array(img[:,:,0]), (1, IMAGE_ROWS, IMAGE_COLS))
	b = np.reshape(np.array(img[:,:,1]), (1, IMAGE_ROWS, IMAGE_COLS))
	c = np.reshape(np.array(img[:,:,2]), (1, IMAGE_ROWS, IMAGE_COLS))
	img1 = np.append(a, b, axis=0)
	toReturn = np.append(img1, c, axis=0)
	return toReturn


# model declaration 
model = Sequential()
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', data_format='channels_first', activation='relu', \
	input_shape=(3,256,256), kernel_initializer='glorot_uniform'))					# output_shape=(,64,256,256)
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))		# output_shape=(,64,256,256)
model.add(MaxPooling2D(pool_size=(2,2)))											# output_shape=(,64,128,128)	paper suggests 4*4

model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))		# output_shape=(,128,128,128)				
model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))		# output_shape=(,128,128,128)
model.add(MaxPooling2D(pool_size=(2,2)))											# output_shape=(,128,64,64)		paper suggests 4*4

model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,256,64,64)
model.add(MaxPooling2D(pool_size=(2,2)))											# output_shape=(,256,32,32)		paper suggests 4*4

model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(2, (1,1), strides=(1,1), padding='same'))							# output_shape=(,2,32,32)
model.add(UpSampling2D(size=(8,8)))													# output_shape=(,2,256,256)
model.add(Lambda(ArgMaxLayer, output_shape=(1,256,256)))							# output_shape=(,1,256,256)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc, mae'])
# inp = np.random.random((16, 3,256,256))
# out_grnd = np.random.random((16, 1,256,256))
# model.fit(inp, out_grnd, batch_size=8, epochs=10, verbose=1)



# make list of all files to be read. then shuffle it
fileNamesArr = []
for fileName in glob.glob(PATH_COLOR + "*.png"):
	fileNamesArr.append(os.path.basename(fileName))
fileNamesArr = random.sample(fileNamesArr, len(fileNamesArr))
print 'done1'



# read images to imageStacks. make nos channels second dimen. scale input to 0-1
c = 0
for eachImgName in fileNamesArr:
	trainInpPath = PATH_COLOR + eachImgName
	trainGrndPath = PATH_HANDMASK + eachImgName
	if (c == 0):
		trainInpStack = makeChannelsSecDimen(trainInpPath).reshape(1, 3, IMAGE_ROWS, IMAGE_COLS)
		trainGrndStack = makeChannelsSecDimen(trainGrndPath).reshape(1, 3, IMAGE_ROWS, IMAGE_COLS)
	else:
		tempInp = makeChannelsSecDimen(trainInpPath).reshape(1, 3, IMAGE_ROWS, IMAGE_COLS)
		tempGrnd = makeChannelsSecDimen(trainGrndPath).reshape(1, 3, IMAGE_ROWS, IMAGE_COLS)
		trainInpStack = np.vstack((trainInpStack, tempInp))
		trainGrndStack = np.vstack((trainGrndStack, tempGrnd))
	c = c+1
trainInpStack = trainInpStack/255
print 'done2'


model.fit(trainInpStack, trainGrndStack, batch_size=8, epochs=10, verbose=1, validation_split=0.2)