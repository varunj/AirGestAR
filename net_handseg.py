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
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K

import theano

import os, glob
import cv2

PATH_COLOR = './data/training/color/'
PATH_HANDMASK = './data/training/mask_hands/'
NOS_INP = 4000

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
	return toReturn/255

def makeChannelsSecDimenGray(path):
	img = cv2.imread(path,0)
	img = cv2.resize(img, (256, 256)) 
	return img

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
model.add(Conv2D(2, (1,1), strides=(1,1), padding='same'))								# output_shape=(,2,32,32)
model.add(UpSampling2D(size=(8,8)))														# output_shape=(,2,256,256)
model.add(Lambda(ArgMaxLayer, output_shape=(1,256,256)))								# output_shape=(,1,256,256)

model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc', 'mae'])
# inp = np.random.random((16, 3,256,256))
# out_grnd = np.random.random((16, 1,256,256))
# model.fit(inp, out_grnd, batch_size=8, epochs=10, verbose=1)



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

model.fit(trainInpStack, trainGrndStack, batch_size=1, epochs=1, verbose=1, validation_split=0.2)

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")


'''
Results

v0.0
NOS_INP=400
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc', 'mae'])
model.fit(trainInpStack, trainGrndStack, batch_size=1, epochs=10, verbose=1, validation_split=0.2)
Train on 320 samples, validate on 80 samples
Epoch 1/10
320/320 [==============================] - 1027s - loss: 3344.5926 - acc: 0.6173 - mean_absolute_error: 13.3582 - val_loss: 2865.0898 - val_acc: 0.6375 - val_mean_absolute_error: 11.4619
Epoch 2/10
320/320 [==============================] - 1054s - loss: 3344.5926 - acc: 0.6173 - mean_absolute_error: 13.3582 - val_loss: 2865.0898 - val_acc: 0.6375 - val_mean_absolute_error: 11.4619
Epoch 3/10
320/320 [==============================] - 1094s - loss: 3344.5926 - acc: 0.6173 - mean_absolute_error: 13.3582 - val_loss: 2865.0898 - val_acc: 0.6375 - val_mean_absolute_error: 11.4619

v1.0
NOS_INP=4000
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc', 'mae'])
model.fit(trainInpStack, trainGrndStack, batch_size=1, epochs=1, verbose=1, validation_split=0.2)
Train on 3200 samples, validate on 800 samples
Epoch 1/1
3200/3200 [==============================] - 10166s - loss: 3201.8065 - acc: 0.6907 - mean_absolute_error: 12.7709 - val_loss: 3356.1294 - val_acc: 0.6889 - val_mean_absolute_error: 13.3814

v1.1
NOS_INP=4000, no /255
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc', 'mae'])
model.fit(trainInpStack, trainGrndStack, batch_size=1, epochs=1, verbose=1, validation_split=0.2)
Train on 3200 samples, validate on 800 samples
Epoch 1/1
Train on 3200 samples, validate on 800 samples
Epoch 1/1
3200/3200 [==============================] - 10085s - loss: 3201.8308 - acc: 0.6247 - mean_absolute_error: 12.7929 - val_loss: 3356.1674 - val_acc: 0.6188 - val_mean_absolute_error: 13.4038
'''