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

nos train = 33k (8/10*41258)
nos iter = 40k
batch size = 8
hence, epochs = 40k/(33k/8) = 10
'''

def doArgMax(inp):
	return K.argmax(inp, axis=1)

model = Sequential()
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', data_format='channels_first', activation='relu', \
	input_shape=(3,256,256), kernel_initializer='glorot_uniform'))					# output_shape=(,64,256,256)
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))		# output_shape=(,64,256,256)
model.add(MaxPooling2D(pool_size=(2,2)))											# output_shape=(,64,128,128)	paper suggests 4*4

model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))		# output_shape=(,128,128,128)				
model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))		# output_shape=(,128,128,128)
model.add(MaxPooling2D(pool_size=(2,2)))											# output_shape=(,128,64,64)	paper suggests 4*4

model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,256,64,64)
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,256,64,64)
model.add(MaxPooling2D(pool_size=(2,2)))											# output_shape=(,256,32,32)	paper suggests 4*4

model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'))		# output_shape=(,512,32,32)
model.add(Conv2D(2, (1,1), strides=(1,1), padding='same'))							# output_shape=(,2,32,32)
model.add(UpSampling2D(size=(8,8)))													# output_shape=(,2,256,256)
model.add(Lambda(doArgMax, output_shape=(1,256,256)))								# output_shape=(,1,256,256)




model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc, mae'])
inp = np.random.random((16, 3,256,256))
out_grnd = np.random.random((16, 1,256,256))
model.fit(inp, out_grnd, batch_size=8, epochs=10, verbose=1)