from __future__ import print_function, unicode_literals
import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import glob, os
import time
from net import ColorHandPose3DNetwork
from utils import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

PATHH = "zimgs"

if __name__ == '__main__':

	# images to be shown
	image_list = list()
	os.chdir("./data/" + PATHH + "")
	for file in sorted(glob.glob("*.png")):
		image_list.append('data/' + PATHH + '/' + file)
	os.chdir("../../")

	# network input
	image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
	hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
	evaluation = tf.placeholder_with_default(True, shape=())

	# build network
	net = ColorHandPose3DNetwork()
	hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
	keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

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

	with open('./weights/weights_Pose3D.pickle', 'rb') as fi:
		weight_dict = pickle.load(fi)
		init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
		sess.run(init_op, init_feed)

	print('now starting on: ' + str(len(image_list)))
	start_time = time.time()

	dic_2d = {}
	dic_3d = {}
	c = 1
	# Feed image list through network
	for img_name in image_list:
		image_raw = scipy.misc.imread(img_name)
		image_raw = scipy.misc.imresize(image_raw, (240, 320))						# input image
		image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0) 		# hand part from input image

		hand_scoremap_v, image_crop_v, scale_v, center_v, keypoints_scoremap_v, keypoint_coord3d_v = \
			sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf], feed_dict={image_tf: image_v})

		hand_scoremap_v = np.squeeze(hand_scoremap_v)
		image_crop_v = np.squeeze(image_crop_v)
		keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
		keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

		# post processing
		image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
		coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
		coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

		# visualize
		# fig = plt.figure()
		# ax1 = fig.add_subplot(221)
		# ax2 = fig.add_subplot(223)
		# ax3 = fig.add_subplot(223)
		# ax4 = fig.add_subplot(224, projection='3d')

		# ax1.imshow(image_raw)
		# plot_hand(coord_hw, ax1)

		# ax2.imshow(image_crop_v)
		# plot_hand(coord_hw_crop, ax2, linewidth='4')

		# ax3.imshow(np.argmax(hand_scoremap_v, 2))

		# plot_hand_3d(keypoint_coord3d_v, ax4)
		# ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
		# ax4.set_xlim([-3, 3])
		# ax4.set_ylim([-3, 1])
		# ax4.set_zlim([-3, 3])


		# fileName = img_name.split('/')[-1].split('.')[0]
		# fig.savefig('./data/yolo_result/' + '_'.join(fileName.split('_')[:-1]) + '_out_' + fileName.split('_')[-1] + '.png', dpi=300, bbox_inches='tight', pad_inches = 0)

		dic_2d[img_name] = coord_hw
		dic_3d[img_name] = keypoint_coord3d_v
		pickle.dump(dic_2d, open( "./result_dics/dic_train_1_2d.pickle", "wb" ) )
		pickle.dump(dic_3d, open( "./result_dics/dic_train_1_3d.pickle", "wb" ) )
		
		print('done: ' + str(c))
		c = c + 1
		# plt.close(fig)
		
	print("--- %s seconds ---" % (time.time() - start_time))