import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)

def viewImage(path):
	img = cv2.imread(path)
	a = np.reshape(np.array(img[:,:,0]), (1, 256, 256))
	b = np.reshape(np.array(img[:,:,1]), (1, 256, 256))
	c = np.reshape(np.array(img[:,:,2]), (1, 256, 256))
	
	print a
	print b
	print c
	print np.amin(img)
	print np.amax(img)

viewImage('aiyo.png')