from __future__ import division 
import tensorflow as tf 
slim = tf.contrib.slim 

##########################################
# Unet: https://arxiv.org/abs/1505.04597 #
##########################################

def unet_arg_scope(weight_decay=1e-4):
  """Defines the Unet arg scope.
    Input: weight_decay - The l2 regularization coefficient
    Output: arg_scope - argument scope of model
    """
  with slim.arg_scope([slim.conv2d],
                      padding='SAME',
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()) as arg_sc:
    return arg_sc

def Unet(inputs,
         is_training=True,
         num_classes = 2,
         dropout_keep_prob=0.775,
         scope='unet'):
    """ Unet 
    Inputs: inputs - input image batch
            is_training - boolean whether to train graph or validate/test
            dropout_keep_prob - probability that each element is kept
            scope - scope name for model
    Outputs: output_map - output logits
             end_points - output dic
    """

    with tf.variable_scope(scope, 'unet', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):

        ######################
        # downsampling  path #
        ######################
        conv1_1 = slim.conv2d(inputs, 32, [3,3], scope='conv1/conv1_1')
        conv1_2 = slim.conv2d(conv1_1, 32, [3,3], scope='conv1/conv1_2')
        pool1 = slim.max_pool2d(conv1_2, [2, 2], scope='pool1')

        conv2_1 = slim.conv2d(pool1, 64, [3,3], scope='conv2/conv2_1')
        conv2_2 = slim.conv2d(conv2_1, 64, [3,3], scope='conv2/conv2_2')
        pool2 = slim.max_pool2d(conv2_2, [2, 2], scope='pool2')

        conv3_1 = slim.conv2d(pool2, 128, [3,3], scope='conv3/conv3_1')
        conv3_2 = slim.conv2d(conv3_1, 128, [3,3], scope='conv3/conv3_2')
        pool3 = slim.max_pool2d(conv3_2, [2, 2], scope='pool3')

        ##############
        # bottleneck #
        ##############
        conv4_1 = slim.conv2d(pool3, 256, [3,3], scope='conv4/conv4_1')
        conv4_2 = slim.conv2d(conv4_1, 256, [3,3], scope='conv4/conv4_2')

        ###################
        # upsampling path #
        ###################
        conv5_1 = slim.conv2d_transpose(conv4_2, 128, [2,2], stride=2, scope ='conv5/transpose_conv5_1')
        merge_1 = tf.concat([conv5_1, conv3_2], axis=-1, name='merge1') # skip connection 1
        merge_1 = slim.dropout(merge_1, dropout_keep_prob, is_training=is_training, scope='dropout1') 
        conv5_2 = slim.conv2d(merge_1, 128, [3,3], scope='conv5/conv5_2')
        conv5_3 = slim.conv2d(conv5_2, 128, [3,3], scope='conv5/conv5_3')

        conv6_1 = slim.conv2d_transpose(conv5_3, 64, [2,2], stride=2, scope='conv6/transpose_conv6_1')
        merge_2 = tf.concat([conv6_1, conv2_2], axis=-1, name='merge2') # skip connection 2
        merge_2 = slim.dropout(merge_2, dropout_keep_prob, is_training=is_training, scope='dropout2')
        conv6_2 = slim.conv2d(merge_2, 64, [3,3], scope='conv6/conv6_2')
        conv6_3 = slim.conv2d(conv6_2, 64, [3,3], scope='conv6/conv6_3')

        conv7_1 = slim.conv2d_transpose(conv6_3, 32, [2,2], stride=2, scope = 'conv7/transpose_conv7_1')
        merge_3 = tf.concat([conv7_1, conv1_2], axis=-1, name='merge3') # skip connection 3
        merge_3 = slim.dropout(merge_3, dropout_keep_prob, is_training=is_training, scope='dropout3')
        conv7_2 = slim.conv2d(merge_3, 32, [3,3], scope='conv7/conv7_2')
        conv7_3 = slim.conv2d(conv7_2, 32, [3,3], scope='conv7/conv7_3')

        ###############
        # outpput map #
        ###############
        output_map = slim.conv2d(conv7_3, num_classes, [1, 1], 
                                activation_fn=None, normalizer_fn=None, 
                                scope='output_layer')

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return output_map, end_points













