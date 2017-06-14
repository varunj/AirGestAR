# place all vids train_bloom_x in data/train_bloomclick_videos. make folder data/train_bloomclick_imgs
import glob, os
import subprocess
PATHH = 'data/train_bloomclick_videos'
PATHHOUT = 'data/train_bloomclick_imgs'
FRAMES = 60

# # make ~60 frames/gest
# arr = []
# for fileName in glob.glob("./" + PATHH + "/*.mp4"):
# 	fileNameSplit = fileName.replace('.', '_').split('_')
# 	fileNamePrefix = 'train_' + fileNameSplit[-3] + '_' + fileNameSplit[-2]
# 	nosFrames = subprocess.check_output("ffprobe -show_streams " + fileName + " | grep \"^nb_frames\" | cut -d '=' -f 2", shell=True)
# 	nosFrames = int(nosFrames[:-1])
# 	step = round(nosFrames/FRAMES,0)
# 	out = subprocess.check_output('ffmpeg -i ' + fileName +  " -vsync 0 -vf \"select='not(mod(n," + str(step) + "))'\" " + PATHHOUT + '/' + fileNamePrefix + "_%d.png", shell=True)
# 	print(step)
# 	print('ffmpeg -i ' + fileName +  " -vsync 0 -vf \"select='not(mod(n," + str(step) + "))'\" " + PATHHOUT + '/' + fileNamePrefix + "_%d.png")
# 	out = subprocess.check_output("ls " + PATHHOUT + " | wc -l", shell=True)
# 	print(out)

# 	arr.append(fileNamePrefix + ' step: ' + str(step) + ' frames: ' + str(nosFrames))
# 	print(arr)
# # now delete all >50


# at 30 fps
arr = []
for fileName in glob.glob("./" + PATHH + "/*.mp4"):
	fileNameSplit = fileName.replace('.', '_').split('_')
	fileNamePrefix = 'train_' + fileNameSplit[-3] + '_' + fileNameSplit[-2]
	out = subprocess.check_output('ffmpeg -i ' + fileName +  " -r 30 " + PATHHOUT + '/' + fileNamePrefix + "_%d.png", shell=True)