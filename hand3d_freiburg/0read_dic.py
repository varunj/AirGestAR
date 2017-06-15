import glob, os
import pickle
import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)

'''
print dics
'''
# for fileName in glob.glob("./result_dics/*2d*_100.pickle"):
# 	dic = pickle.load(open(fileName, "rb" ))
# 	for x,y in dic.items():
# 		print(x, y.shape)
	# pprint.pprint(dic)

'''
show histogram of len of gestures from dic
'''
# for fileName in glob.glob("./result_dics/*2d*.pickle"):
# 	dic = pickle.load(open(fileName, "rb" ))
# 	dicHist = {}
# 	for gestureNames in dic.keys():
# 		gestureNameSmall = gestureNames.split('_')[-3] + gestureNames.split('_')[-2]
# 		if not gestureNameSmall in dicHist:
# 			dicHist[gestureNameSmall] = 1
# 		else:
# 			dicHist[gestureNameSmall] += 1			
# pprint.pprint(dicHist)
# pprint.pprint(len(dicHist))

'''
show histogram of len of gestures from folder of images
'''
# dicHist = {}
# for fileName in glob.glob("./data/train_zoom_imgs/*.png"):
# 	gestureNameSmall = fileName.split('_')[-3] + fileName.split('_')[-2]
# 	if not gestureNameSmall in dicHist:
# 		dicHist[gestureNameSmall] = 1
# 	else:
# 		dicHist[gestureNameSmall] += 1
# pprint.pprint(dicHist)
# pprint.pprint(len(dicHist))

'''
resample to 100/gesture instance
'''
# TARGET_LEN = 100

# def addBetween(inpList, x):
# 	a = np.zeros(shape=(1,21,2))
# 	for y in range(0,21):
# 		a[0,y,0] = (inpList[x-1][y][0]+inpList[x][y][0])/2.0
# 		a[0,y,1] = (inpList[x-1][y][1]+inpList[x][y][1])/2.0
# 	toReturn = np.append(inpList[:x,:,:], a, axis=0)
# 	toReturn = np.append(toReturn, inpList[x:,:,:], axis=0)
# 	return toReturn

# for fileName in glob.glob("./result_dics/*2d*.pickle"):
# 	dic = pickle.load(open(fileName, "rb" ))
# 	dicGesture = {}

# 	gestureCount = [str(x) for x in range(1,26)]
# 	gestureSeq = [str(x) for x in range(1,200)]
# 	gesture = ['bloom', 'click']
# 	for ges in gesture:
# 		for gesC in gestureCount:
# 			for gesS in gestureSeq:
# 				if ('data/train_bloomclick_imgs/train_' + ges + '_' + gesC + '_' + gesS + '.png' in dic.keys()):
# 					if (ges+gesC not in dicGesture):
# 						dicGesture[ges+gesC] = [dic['data/train_bloomclick_imgs/train_' + ges + '_' + gesC + '_' + gesS + '.png']]
# 					else:
# 						dicGesture[ges+gesC].append(dic['data/train_bloomclick_imgs/train_' + ges + '_' + gesC + '_' + gesS + '.png'])
# # print len of seqs
# for x,y in dicGesture.items():
# 	print(x, len(y))

# # convert to numpy
# for x,y in dicGesture.items():
# 	dicGesture[x] = np.asarray(y)

# # resample
# for gesture, coordinateList in dicGesture.items():
# 	inplen = len(coordinateList)

# 	if (inplen < TARGET_LEN):
# 		x = 1
# 		while (inplen < TARGET_LEN):
# 			dicGesture[gesture] = addBetween(dicGesture[gesture], x)
# 			x = x + 2
# 			if (x == len(dicGesture[gesture])):
# 				x = 1
# 			inplen = len(dicGesture[gesture])

# 	else:
# 		listRem = []
# 		for x in range(1, len(dicGesture[gesture])):
# 			if (x%2 == 0):
# 				listRem.append(x)
# 		listRem = listRem[:min(len(listRem), abs(TARGET_LEN-len(dicGesture[gesture])))]
# 		dicGesture[gesture] = np.delete(dicGesture[gesture], listRem, axis=0)

# # print len of seqs
# for x,y in dicGesture.items():
# 	print(x, len(y))

# for fileName in glob.glob("./result_dics/*2d*.pickle"):
# 	pickle.dump(dicGesture, open(fileName + '_resampled', "wb" ) )