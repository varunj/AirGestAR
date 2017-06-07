import os
import pandas as pd
import numpy as np
import pdb
import cv2
img_rows, img_cols = 99,99
pdb.set_trace()

Path="/home/ramakrishna/Documents/kerasCodes/egofing_gaurav/txt/"
Path2="/home/ramakrishna/Documents/kerasCodes/egofing_gaurav/CSVlabels/"
Path1="/home/ramakrishna/Documents/DeepLearning/py-faster-rcnn/data/VOCdevkit2012/VOC2012/JPEGImages/"
Path3="/home/ramakrishna/Documents/kerasCodes/egofing_gaurav/CroppedImages/"
for filename in os.listdir(Path):
        if filename.endswith(".txt"):
	   print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
           print filename.format(filename)
           data = pd.read_csv(Path+filename, header = None, delim_whitespace = True) # annotations file path with delimiter
	   data.ix[:,1]=data.ix[:,1]*640
	   data.ix[:,2]=data.ix[:,2]*480
           data.ix[:,3]=data.ix[:,3]*640
	   data.ix[:,4]=data.ix[:,4]*480
	   data.ix[:,5]=data.ix[:,5]*640
	   data.ix[:,6]=data.ix[:,6]*480
   	   data.ix[:,7]=data.ix[:,7]*640
	   data.ix[:,8]=data.ix[:,8]*480
	   
	   s=len(data)

	   width=data.ix[:,3]-data.ix[:,1]
           height=data.ix[:,4]-data.ix[:,2]
	   
	   for i in range(0,s):
		img = cv2.imread(Path1+data.ix[i,0])	
		crop_img = img[int(round(data.ix[i,2])):int(round(data.ix[i,2]+height[i])), int(round(data.ix[i,1])):int(round(data.ix[i,1]+width[i]))] # Crop from x, y, w, h -> 100, 200, 300, 400
	        im = cv2.resize(crop_img, (img_rows, img_cols)).astype(np.float32)  
    		cv2.imwrite(Path3 +'/' +  str(data.ix[i,0][:-4])+ "_crop.png" , im)
  
	   print "data resizing has been done"	

	   data.ix[:,5]=data.ix[:,5]-data.ix[:,1]
	   data.ix[:,6]=data.ix[:,6]-data.ix[:,2]
	   data.ix[:,7]=data.ix[:,7]-data.ix[:,1]
	   data.ix[:,8]=data.ix[:,8]-data.ix[:,2]
	   data.ix[:,5]=(data.ix[:,5]*img_cols)/width
	   data.ix[:,6]=(data.ix[:,6]*img_rows)/height
	   data.ix[:,7]=(data.ix[:,7]*img_cols)/width
	   data.ix[:,8]=(data.ix[:,8]*img_rows)/height

	   labels = np.zeros(shape=(s,4))
	   labels[:,0] = data.ix[:,5]
	   labels[:,1] = data.ix[:,6]
	   labels[:,2] = data.ix[:,7]
	   labels[:,3] = data.ix[:,8]
	   
	   raw_data = {'X1': labels[:,0],
           'X2': labels[:,1],
           'X3': labels[:,2],
           'X4': labels[:,3],
           'Image': data.ix[:,0]}
	   df = pd.DataFrame(raw_data, columns = ['X1', 'X2', 'X3', 'X4', 'Image'])
           df.to_csv(Path2+filename[0:-4]+'.csv', index=False, header =False)
           
	else:
	   print "no txt file is there" 
	
