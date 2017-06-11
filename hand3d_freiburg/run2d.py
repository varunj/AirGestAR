#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import glob, os
import time

from net2d import ColorHandPose3DNetwork
from utils import detect_keypoints, trafo_coords, plot_hand

if __name__ == '__main__':

	# images to be shown
	image_list = list()
	os.chdir("./data")
	for file in glob.glob("*.png"):
		image_list.append('data/' + file)
	os.chdir("../")

	# network input
	image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
	hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
	evaluation = tf.placeholder_with_default(True, shape=())

	# build network
	net = ColorHandPose3DNetwork()
	hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
	keypoints_scoremap_tf = net.inference(image_tf, hand_side_tf, evaluation)

	# Start TF
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	# init = tf.initialize_all_variables()
	# sess.run(init)

	# Initialize with weights
	with open('./weights/weights_HandSegNet.pickle', 'rb') as fi:
		weight_dict = pickle.load(fi)
		init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
		sess.run(init_op, init_feed)

	with open('./weights/weights_Pose2D.pickle', 'rb') as fi:
		weight_dict = pickle.load(fi)
		init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
		sess.run(init_op, init_feed)

	print('Now Starting!!!')
	# start_time = time.time()

	# Feed image list through network
	for img_name in image_list:
		image_raw = scipy.misc.imread(img_name)
		image_raw = scipy.misc.imresize(image_raw, (240, 320))						# input image
		image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

		hand_scoremap_v, image_crop_v, scale_v, center_v, keypoints_scoremap_v = \
			sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf], feed_dict={image_tf: image_v})

		hand_scoremap_v = np.squeeze(hand_scoremap_v)								# hand seg mask
		image_crop_v = np.squeeze(image_crop_v)										# cropped image
		keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)						# long info of 21 pts???

		# post processing
		image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
		coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))			# 21x2 vector of 21 <x,y> pts in cropped
		coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)				# 21x2 vector of 21 <x,y> pts in global

		# visualize
		fig = plt.figure()
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)

		# original hand + 21 pts
		ax1.imshow(image_raw)
		plot_hand(coord_hw, ax1)

		# cropped hand + 21 pts
		ax2.imshow(image_crop_v)
		plot_hand(coord_hw_crop, ax2)

		# segmented hand
		ax3.imshow(np.argmax(hand_scoremap_v, 2))

		fileName = img_name.split('/')[-1].split('.')[0]
		fig.savefig('result/' + fileName.split('_')[0] + '_out_' + fileName.split('_')[1] + '.png')
		print('done' + fileName)

		plt.close(fig)
	# print("--- %s seconds ---" % (time.time() - start_time))