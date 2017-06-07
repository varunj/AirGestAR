#THEANO_FLAGS=device=cuda0 python steps.py
import os 
#os.environ['THEANO_FLAGS']="device=gpu"
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import theano
import keras
#KERAS
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

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

import cv2
# input image dimensions
img_rows, img_cols = 99,99

# number of channels
img_channels = 1
#data

'''
model = Sequential()
model.add(Convolution2D( 32, (5,5),strides=(2, 2), input_shape=(3,99,99), border_mode='same', activation='relu', init='uniform', W_regularizer=l2(0.01) ))
model.add(Convolution2D( 32, (5,5), border_mode='same', activation='relu', init='uniform', W_regularizer=l2(0.01) ))
model.add(MaxPooling2D( pool_size=(2,2) ))
model.add(Convolution2D( 64, (5,5),strides=(2, 2), border_mode='same', activation='relu', init='uniform', W_regularizer=l2(0.01) ))
model.add(Convolution2D( 64, (5,5), border_mode='same', activation='relu', init='uniform', W_regularizer=l2(0.01) ))
model.add(MaxPooling2D( pool_size=(2,2) ))
model.add(Flatten())
model.add(Dense( 160, activation='relu', init='uniform', W_regularizer=l2(0.01) ))
model.add(Dense( 4, init='uniform', W_regularizer=l2(0.01) ))
model.compile(loss="mean_squared_error", optimizer="adam")
'''

data_path = '/home/ramakrishna/Documents/kerasCodes/point_data/'  #path of folder to save images 
		
#gen_annotations = pd.read_csv('/home/ramakrishna/Documents/kerasCodes/point_anno/I_Avenue_label.csv', sep=",", header = None)
def make_chan_first(path):
	img=cv2.imread(path)
	a = np.array(img[:,:,0])
	#pdb.set_trace()
	b= np.array(img[:,:,1])
	c= np.array(img[:,:,2])
	a= np.reshape(a, (1,img_rows,img_cols))
	b= np.reshape(b, (1,img_rows,img_cols))
	c= np.reshape(c, (1,img_rows,img_cols))
	img1 = np.append(a,b, axis=0)
	chan_fst_img = np.append(img1, c, axis =0)
	return chan_fst_img

#pdb.set_trace()

#im1 = make_chan_first('/home/ramakrishna/Pictures/fram1.jpg')
#imgstack = im1.reshape(1,3,99,99)

#orig_annotations = pd.read_csv('/home/ramakrishna/Documents/kerasCodes/txt/I_Avenue_label.txt', sep="    ", header = None)
data_list = ['I_TeachingBuilding/']
ann1 = '/home/ramakrishna/Documents/kerasCodes/txt/I_TeachingBuilding.txt'
ann2= '/home/ramakrishna/Documents/kerasCodes/point_anno/I_TeachingBuilding.csv'
count =0
cnt =0 
#labels = np.zeros(shape=(4058,4))
orig_annotations = pd.read_csv(ann1, delim_whitespace = True, header = None)
gen_annotations = pd.read_csv(ann2, sep=",", header = None)
labels = np.zeros(shape = (len(orig_annotations),4))
pdb.set_trace()
print "image data loading ..."
for k in range(1): # make it 10 later
	print "k value is",k
	data_path2 = data_path + data_list[k]
	#orig_annotations = pd.read_csv(ann1, delim_whitespace = True, header = None)
	#gen_annotations = pd.read_csv(ann2, sep=",", header = None)
	if (cnt ==0):
		labels[:,0:4] = gen_annotations.ix[:,0:4]
		cnt = cnt+1
	else:	
		labels = np.append(labels, gen_annotations.ix[:,0:4],axis =0)
	#pdb.set_trace()
	for img in orig_annotations.ix[:,0]:
		img = img.split('.png')[0]+'_crop.png'
		if (count ==0):
			im1 = make_chan_first(data_path2+img) # open one image to get size
			imgstack=im1.reshape(1,3,img_rows,img_cols)
			count =count+1
		else:
			im2 = make_chan_first(data_path2+img) # open one image to get size
			#print im1.shape, img
			im3=im2.reshape(1,3,img_rows,img_cols)
			imgstack= np.vstack((imgstack, im3))

#pdb.set_trace()
#images normalization
print "image normalization process..."
imgs = np.float32(imgstack)
img_data = imgs/255 # bring to scale of 0-1
#img_mean= np.mean(img_data,axis=0)
'''
img_mean = np.load('params/mean_img.npy')

#img_std = np.std(img_data,axis=0)
img_std = np.load('params/std_img.npy')
img_data1 = img_data - img_mean # sub mean
img_data_norm = img_data1 / img_std # divide with std deviation
print "img data has been loaded "

#model = load_model('weights.best.hdf5')
pdb.set_trace()
'''
print "model prediction process ..."
model = load_model('weights.best_point_model12.hdf5')
scores = model.predict(img_data,verbose =1)
diff= scores - labels 
diffabs = np.abs(diff)
total_error = np.mean(diffabs)
print ("Mean Absolute Error: %.2f%% pixels" % total_error)
np.savetxt('resultsI_NcanteenShftemp12.csv', scores, delimiter=',')
pdb.set_trace()
#labels normalization (x- min )/max for corresponding label
#min_values = np.min(labels, axis=0)
#max_values = np.max(labels1, axis =0)
'''
max_values=np.load('params/max_labels.npy')
scores1 = scores * max_values
min_values=np.load('params/min_labels.npy')
labels = scores1 + min_values
np.savetxt('resultsNorthCanteen.csv', labels, delimiter=',')
pdb.set_trace()'''
