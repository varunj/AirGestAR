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

import tensorflow as tf
from utils import *
ops = NetworkOps

class ColorHandPose3DNetwork(object):
    """ Network performing 3D pose estimation of a human hand from single color image. """
    def __init__(self):
        self.crop_size = 256
        self.num_kp = 21

    def inference(self, image, hand_side, evaluation):
        # use network for hand segmentation for detection
        hand_scoremap = self._inference_detection(image)

        # Intermediate data processing
        hand_mask = single_obj_scoremap(hand_scoremap)
        center, _, crop_size_best = calc_center_bb(hand_mask)
        crop_size_best *= 1.25
        scale_crop = tf.minimum(tf.maximum(self.crop_size / crop_size_best, 0.25), 5.0)
        image_crop = crop_image_from_xy(image, center, self.crop_size, scale=scale_crop)

        # detect keypoints in 2D
        keypoints_scoremap = self._inference_pose2d(image_crop)
        
        # upsample keypoint scoremap
        s = image_crop.get_shape().as_list()
        keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (s[1], s[2]))

        return hand_scoremap, image_crop, scale_crop, center, keypoints_scoremap

    @staticmethod
    def _inference_detection(image, train=False):
        """ 
            Detects the hand in the input image by segmenting it. 
            block_id: 1-4, corresponding to layers 1-3, 4-6, 7-11, 12-15
            layer_id: convlayer# within a block. blk1(0,1), blk2(0,1), blk3(0,1,2,3), blk4(0,1,2,3)
            kernel: 3x3 for all
            stride: 1x1 for all
            chan_num: filter size
        """
        with tf.variable_scope('HandSegNet'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 4]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # learn some feature representation, that describes the image content well. 
            # layer 1-15
            x = image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)
            # layer 16
            x = ops.conv_relu(x, 'conv5_1', kernel_size=3, stride=1, out_chan=512, trainable=train)
            
            # use encoding to detect initial scoremap. 
            # layer??
            encoding = ops.conv_relu(x, 'conv5_2', kernel_size=3, stride=1, out_chan=128, trainable=train)
            x = ops.conv_relu(encoding, 'conv6_1', kernel_size=1, stride=1, out_chan=512, trainable=train)

            # layer 17
            scoremap = ops.conv(x, 'conv6_2', kernel_size=1, stride=1, out_chan=2, trainable=train)
            scoremap_list.append(scoremap)

            # upsample to full size/ uses Bilinear interpolation
            s = image.get_shape().as_list()
            scoremap_list_large = [tf.image.resize_images(x, (s[1], s[2])) for x in scoremap_list]
            
        return scoremap_list_large[-1]

    def _inference_pose2d(self, image_crop, train=False):
        """ Given an image it detects the 2D hand keypoints. """
        with tf.variable_scope('PoseNet2D'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 2]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # learn some feature representation, that describes the image content well
            x = image_crop
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    print('---layer_id:' + str(layer_id))
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            x = ops.conv_relu(x, 'conv4_3', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_4', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_5', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_6', kernel_size=3, stride=1, out_chan=256, trainable=train)
            encoding = ops.conv_relu(x, 'conv4_7', kernel_size=3, stride=1, out_chan=128, trainable=train)

            # use encoding to detect initial scoremap
            x = ops.conv_relu(encoding, 'conv5_1', kernel_size=1, stride=1, out_chan=512, trainable=train)
            scoremap = ops.conv(x, 'conv5_2', kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
            scoremap_list.append(scoremap)

            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 5
            num_recurrent_units = 2
            for pass_id in range(num_recurrent_units):
                # layer 18, 25
                x = tf.concat(3, [scoremap_list[-1], encoding])
                for rec_id in range(layers_per_recurrent_unit):
                    # layer 19-23, 26-30
                    x = ops.conv_relu(x, 'conv%d_%d' % (pass_id+6, rec_id+1), kernel_size=7, stride=1, out_chan=128, trainable=train)
                # layer??
                x = ops.conv_relu(x, 'conv%d_6' % (pass_id+6), kernel_size=1, stride=1, out_chan=128, trainable=train)
                # layer 24, 31
                scoremap = ops.conv(x, 'conv%d_7' % (pass_id+6), kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
                scoremap_list.append(scoremap)

            scoremap_list_large = scoremap_list

        return scoremap_list_large[-1]