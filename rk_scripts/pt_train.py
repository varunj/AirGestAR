#THEANO_FLAGS=device=cuda0 python steps.py
import time
start_time = time.time()
import os 
#os.environ['THEANO_FLAGS']="device=gpu"
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import theano
import keras
import keras.backend as K

#KERAS
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, Callback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import pdb
from pylab import figure, axes, pie, title, show
import scipy.misc 
import cv2

# input image dimensions
img_rows, img_cols = 99,99

# -----------------model declaration

Image_path = '/home/ramakrishna/Documents/kerasCodes/egofing_gaurav/CroppedImages/'  #path of folder to save images 
Path="/home/ramakrishna/Documents/kerasCodes/egofing_gaurav/CSVlabels/"



labels = []
count =0
cnt =0 
for filename in os.listdir(Path):
        if filename.endswith(".csv"):
	   print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
           print filename.format(filename)
           gen_annotations = pd.read_csv(Path+filename, header = None, sep=",") # annotations file path with delimiter
 	   if (cnt ==0):
		lab = gen_annotations.ix[:,0:5]
		#gen_annotations=shuffle_data(len(gen_annotations),gen_annotations)
		labels = np.array(lab)
		cnt = cnt+1
	   else:	
		labels = np.append(labels, gen_annotations.ix[:,0:5],axis =0)


labels=shuffle_data(len(labels),labels)
print("---time upto Shuffling time %s seconds ---" % (time.time() - start_time))
labels2=np.float16(labels[:,0:4])
pdb.set_trace()
im_time1=time.time() 
for img in labels[:,4]:
		im_time=time.time() 
		img = img.split('.png')[0]+'_crop.png'
		print img,"   count: ",count
		if (count ==0):
			im1 = make_chan_first(Image_path + img)
			im1 = np.float16(im1)
			imgstack=im1.reshape(1,3,img_rows,img_cols)
			count =count+1
		else:
			im2 = make_chan_first(Image_path + img)
			im2 = np.float16(im2)
			im3 = im2.reshape(1,3,img_rows,img_cols)
			imgstack= np.vstack((imgstack, im3))
			count =count+1
		print("---time to load one image is time %s seconds ---" % (time.time() - im_time))
print("---total time to load one image is time %s seconds ---" % (time.time() - im_time1))
#images normalization
imgs = np.float16(imgstack)
img_data = imgs/255 # bring to scale of 0-1

#labels2=np.float32(labels2[:,0:4])


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        print 'hi'
        print "predicted values are", model.predict(img_data[100:105,:,:,:])
        print "actual labels are", labels[100:105,:]

history = LossHistory()
filepath="weights.best_point_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]


history = model.fit(img_data, labels2, nb_epoch=50, batch_size=100, validation_split=0.25, callbacks=callbacks_list, verbose = 1)
#model.fit(rand_data, rand_labels, nb_epoch=50, batch_size=60, validation_split=0.2, callbacks=callbacks_list, verbose = 1)
print("---time to do training one time  %s seconds ---" % (time.time() - start_time))
pdb.set_trace()
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('gargacc1.png',dpi=100	)
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('c33labNloss.png',dpi=100)
plt.show()
print("---time to do complete training in %s seconds ---" % (time.time() - start_time))