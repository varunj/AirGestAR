# THEANO_FLAGS=device=gpu,floatX=float32 python net_handseg_deconv.py
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
NOS_INP = 4000
EPOCHS_NO = 3
BATCH_SIZE = 8

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

model.add(Conv2DTranspose(256, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer='glorot_uniform'	))		# output_shape=(,256,64,64)
model.add(Conv2DTranspose(128, (3,3), strides=(2,2), border_mode='same', activation='relu', kernel_initializer='glorot_uniform'))	# output_shape=(,128,128,128)
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer='glorot_uniform'))			# output_shape=(,1,256,256)

# opt = SGD(lr=0.00001)
# model.compile(loss = "categorical_crossentropy", optimizer = opt)
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
with open("model2.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model2.h5")


'''
Results

v2.0 BLACK
NOS_INP=4000
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc', 'mae'])
model.fit(trainInpStack, trainGrndStack, batch_size=8, epochs=3, verbose=1, validation_split=0.2)
Train on 3200 samples, validate on 800 samples
Epoch 1/3
3200/3200 [==============================] - 295s - loss: 0.0475 - acc: 0.1810 - absolute_error: 0798 - val_loss: 0.0516 - val_acc: 0.6912 - val_mean_absolute_error: 0.0525 0.06
Epoch 2/3
3200/3200 [==============================] - 200s - loss: 0.0492 - acc: 0.6942 - absolute_error: 0.0501 - val_loss: 0.0516 - val_acc: 0.6912 - val_mean_absolute_e.0525506
Epoch 3/3
3200/3200 [==============================] - 242s - loss: 0.0492 - acc: 0.6942 - absolute_error: 0501 - val_loss: 0.0516 - val_acc: 0.6912 - val_mean_absolute_error: 0.0525

'''