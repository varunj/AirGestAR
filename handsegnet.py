import numpy as np
np.random.seed(123)

from keras.models import Sequential	
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda
from keras.utils import np_utils
from keras import backend as K
import theano
'''
keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, 
kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

Conv2D(nos of conv filters, (dimen of each filter/kernel), input_shape=(depth, width, height))
'''

def doArgMax(inp):
	argmaxedInp = K.argmax(inp, axis=0)
	return argmaxedInp

model = Sequential()
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', data_format='channels_first', activation='relu', \
	input_shape=(3,256,256), kernel_initializer='glorot_uniform'))					# output_shape=(64,256,256)
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))		# output_shape=(64,256,256)
model.add(MaxPooling2D(pool_size=(2,2)))											# output_shape=(64,128,128)	paper suggests 4*4

model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))		# output_shape=(128,128,128)				
model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))		# output_shape=(128,128,128)
model.add(MaxPooling2D(pool_size=(2,2)))											# output_shape=(128,64,64)	paper suggests 4*4

model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(256,64,64)
model.add(MaxPooling2D(pool_size=(2,2)))											# output_shape=(256,32,32)	paper suggests 4*4

model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(512,32,32)
model.add(Conv2D(2, (1,1), strides=(1,1), padding='same'))							# output_shape=(2,32,32)
model.add(UpSampling2D(size=(8,8)))													# output_shape=(2,256,256)


print model.layers[-1].output_shape
model.add(Lambda(doArgMax, output_shape=(1,256,256)))
print model.layers[-1].output_shape


model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc'])
inp = np.random.random((16, 3,256,256))
out_grnd = np.random.random((16, 1,256,256))
model.fit(inp, out_grnd, batch_size=8, epochs=40000, verbose=1)