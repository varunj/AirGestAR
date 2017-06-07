#THEANO_FLAGS=device=cuda0 python steps.py
#KERAS
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.regularizers import l2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
import h5py
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import pdb
import cv2
#os.environ['THEANO_FLAGS']="device=gpu"
import theano

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

# input image dimensions
img_rows, img_cols = 99,99

path1 = '/home/ramakrishna/Documents/kerasCodes/point_data/I_TeachingBuilding'    #path of folder of images
orig_ann =   pd.read_csv('/home/ramakrishna/Documents/kerasCodes/txt/I_TeachingBuilding.txt', delim_whitespace = True, header= None)     # orig annotations txt file
pred = np.loadtxt('/home/ramakrishna/Documents/kerasCodes/resultsI_NcanteenShftemp12.csv', delimiter = ',') #results csv file to be seen 
#pred = np.loadtxt('/home/ramakrishna/Documents/kerasCodes/point_anno/I_NorthCanteen_label.csv', delimiter = ',') 

pred = pred.astype(int)
pdb.set_trace()
for i in range(0,len(pred)):
    img = cv2.imread(path1+ "/"+ str(orig_ann.ix[i,0][:-4])+ "_crop.png")
    print str(orig_ann.ix[i,0][:-4])
    cv2.circle(img, (pred[i,0],pred[i,1]), 3, 1234, thickness=1, lineType=8, shift=0)
    cv2.circle(img, (pred[i,2],pred[i,3]), 3, 1234, thickness=1, lineType=8, shift=0)
    cv2.imshow("images ",img)
    cv2.waitKey(100)
  
print "completed visulazing images"	
