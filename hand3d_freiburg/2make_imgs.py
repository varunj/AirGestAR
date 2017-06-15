# place all vids data/train_bloomclick_videos. make folder data/train_bloomclick_imgs
import glob, os
import subprocess
PATHHIN = 'data/train_zoom_videos'
PATHHOUT = 'data/train_zoom_imgs'
FRAMES = 60

# at 30 fps
arr = []
for fileName in glob.glob("./" + PATHHIN + "/*.mp4"):
	fileNameSplit = fileName.replace('.', '_').split('_')
	fileNamePrefix = 'train_' + fileNameSplit[-3] + '_' + fileNameSplit[-2]
	out = subprocess.check_output('ffmpeg -i ' + fileName +  " -r 30 " + PATHHOUT + '/' + fileNamePrefix + "_%d.png", shell=True)