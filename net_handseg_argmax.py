# THEANO_FLAGS=device=gpu,floatX=float32 python net_handseg.py
import theano

import numpy as np
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

PATH_COLOR = './data/training/color/'
PATH_HANDMASK = './data/training/mask_hands/'
NOS_INP = 200
EPOCHS_NO = 3
BATCH_SIZE = 1

def ArgMaxLayer(inp):
	return K.argmax(inp, axis=1)

def makeChannelsSecDimen(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (256, 256)) 
	a = np.reshape(np.array(img[:,:,0]), (1, 256, 256))
	b = np.reshape(np.array(img[:,:,1]), (1, 256, 256))
	img1 = np.append(a, b, axis=0)
	c = np.reshape(np.array(img[:,:,2]), (1, 256, 256))
	toReturn = np.append(img1, c, axis=0)
	return toReturn/255.0

def makeChannelsSecDimenGray(path):
	img = cv2.imread(path,0)
	img[img > 0] = 255
	img = cv2.resize(img, (256, 256)) 
	return img/255.0

# model declaration 
model = Sequential()
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', data_format='channels_first', activation='relu', \
	input_shape=(3,256,256), kernel_initializer='glorot_uniform'))						# output_shape=(,64,256,256)
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))			# output_shape=(,64,256,256)
model.add(MaxPooling2D(pool_size=(2,2)))												# output_shape=(,64,128,128)	paper suggests 4*4

model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))			# output_shape=(,128,128,128)				
model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))			# output_shape=(,128,128,128)
model.add(MaxPooling2D(pool_size=(2,2)))												# output_shape=(,128,64,64)		paper suggests 4*4

model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,256,64,64)
model.add(MaxPooling2D(pool_size=(2,2)))												# output_shape=(,256,32,32)		paper suggests 4*4

model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,512,32,32)

model.add(Conv2D(2, (1,1), strides=(1,1), activation='relu', padding='same'))			# output_shape=(,2,32,32)
model.add(UpSampling2D(size=(8,8)))														# output_shape=(,2,256,256)
model.add(Lambda(ArgMaxLayer, output_shape=(1,256,256)))								# output_shape=(,1,256,256)

# opt = SGD(lr=0.00001)
# model.compile(loss = "binary_crossentropy", optimizer = opt)
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc', 'mae'])
# inp = np.random.random((16, 3,256,256))
# out_grnd = np.random.random((16, 1,256,256))
# model.fit(inp, out_grnd, batch_size=1, epochs=10, verbose=1)


# make list of all files to be read. then shuffle it
fileNamesArr = []
for fileName in glob.glob(PATH_COLOR + "*.png"):
	fileNamesArr.append(os.path.basename(fileName))
fileNamesArr = random.sample(fileNamesArr, len(fileNamesArr))

# read images to imageStacks. make nos channels second dimen. scale input to 0-1
c = 0
trainInpStack = np.zeros((NOS_INP, 3, 256, 256))
trainGrndStack = np.zeros((NOS_INP, 1, 256, 256))

for eachImgName in fileNamesArr[:NOS_INP]:
	trainInpPath = PATH_COLOR + eachImgName
	trainGrndPath = PATH_HANDMASK + eachImgName
	trainInpStack[c] = makeChannelsSecDimen(trainInpPath).reshape(1, 3, 256, 256)
	trainGrndStack[c] = makeChannelsSecDimenGray(trainGrndPath).reshape(1, 1, 256, 256)
	c = c+1
	print 'done for image #' + str(c)

model.fit(trainInpStack, trainGrndStack, batch_size=BATCH_SIZE, epochs=EPOCHS_NO, verbose=1, validation_split=0.2)

model_json = model.to_json()
with open("model0.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model0.h5")


'''
Results

v0.0
NOS_INP=4000
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc', 'mae'])
model.fit(trainInpStack, trainGrndStack, batch_size=1, epochs=3, verbose=1, validation_split=0.2)
Train on 3200 samples, validate on 800 samples
Epoch 1/3
3200/3200 [==============================] - 1720s - loss: 0.0700 - acc: 0.6327 - mean_absolute_error: 0.0700 - val_loss: 0.0726 - val_acc: 0.6263 - val_mean_absolute_error: 0.0726
Epoch 2/3
3200/3200 [==============================] - 595s - loss: 0.0700 - acc: 0.6327 - mean_absolute_error: 0.0700 - val_loss: 0.0726 - val_acc: 0.6263 - val_mean_absolute_error: 0.0726
Epoch 3/3
3200/3200 [==============================] - 249s - loss: 0.0700 - acc: 0.6327 - mean_absolute_error: 0.0700 - val_loss: 0.0726 - val_acc: 0.6263 - val_mean_absolute_error: 0.0726

'''