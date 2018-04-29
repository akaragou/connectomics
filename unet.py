from __future__ import division 
import tensorflow as tf 
slim = tf.contrib.slim 

def conv3d(net, output_dim, f_size, is_training, layer_name):
    with tf.variable_scope(layer_name):

        w = tf.get_variable('w', [f_size, f_size, f_size, net.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv3d(net, w, strides=[1,1,1,1], padding='SAME')
        b = tf.get_variable('b', [output_dim], initializer=tf.costant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        bn = tf.contrib.layers.batch_norm(conv, is_training=is_training, scope='bn',
                                 decay=0.997, epsilon=1e-5, center=True, scale=True)

        r = tf.nn.elu(bn)
        return r

def deconv3d(net, output_shape, f_size, is_training, layer_name):
    with tf.variable_scope(layer_name):
        w = tf.get_variable('w', [f_size, f_size, f_size, output_shape[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        deconv = tf.nn.conv3d_transpose(net, w, output_shape, strides=[1, f_size, f_size, f_size, 1], padding='SAME')
        bn = tf.contrib.layers.batch_norm(conv, is_training=is_training, scope='bn',
                                 decay=0.997, epsilon=1e-5, center=True, scale=True)
        r = tf.nn.elu(bn)
        return r

def conv_residual_block(net, num_features, is_training, is_batch_norm, layer_name):
    """ Unet_V2 residual conv block 
        Inputs: net - input feature map
                num_features - number of features in convolution block
                is_training - boolean whether to train graph or validate/test
                is_batch_norm - boolean whether to have batchnorm activated or not
                layer_name - scope name for layer

        Output: net - a feature map 
    """
    with tf.variable_scope(layer_name):
        net = slim.conv2d(net, num_features, [3,3], activation_fn=None,  normalizer_fn=None, scope='conv%d_1' % int(layer_name[-1]))
        if is_batch_norm:
            net = slim.batch_norm(net, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='batch_norm1')
        shortcut = tf.nn.elu(net)
        net = shortcut
        net = slim.conv2d(net, num_features, [3,3], activation_fn=None,  normalizer_fn=None, scope='conv%d_2' % int(layer_name[-1]))
        if is_batch_norm:
            net = slim.batch_norm(net, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='batch_norm2')
        net = tf.nn.elu(net)
        net = slim.conv2d(net, num_features, [3,3], activation_fn=None,  normalizer_fn=None, scope='conv%d_3' % int(layer_name[-1]))
        if is_batch_norm:
            net = slim.batch_norm(net, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='batch_norm3')
        net = tf.nn.elu(net + shortcut)
        return net


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

##########################################
# Unet: https://arxiv.org/abs/1505.04597 #
##########################################

def Unet(inputs,
         is_training=True,
         num_classes = 2,
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
        conv1_1 = slim.conv2d(inputs, 64, [3,3], scope='conv1/conv1_1')
        conv1_2 = slim.conv2d(inputs, 64, [3,3], scope='conv1/conv1_2')
        pool1 = slim.max_pool2d(conv1_2, [2, 2], scope='pool1')

        conv2_1 = slim.conv2d(pool1, 128, [3,3], scope='conv2/conv2_1')
        conv2_2 = slim.conv2d(conv2_1, 128, [3,3], scope='conv2/conv2_2')
        pool2 = slim.max_pool2d(conv2_2, [2, 2], scope='pool2')

        conv3_1 = slim.conv2d(pool2, 256, [3,3], scope='conv3/conv3_1')
        conv3_2 = slim.conv2d(conv3_1, 256, [3,3], scope='conv3/conv3_2')
        pool3 = slim.max_pool2d(conv3_2, [2, 2], scope='pool3')

        conv4_1 = slim.conv2d(pool3, 512, [3,3], scope='conv4/conv4_1')
        conv4_2 = slim.conv2d(conv4_1, 512, [3,3], scope='conv4/conv4_2')
        pool4 = slim.max_pool2d(conv4_2, [2, 2], scope='pool4')

        ##############
        # bottleneck #
        ##############
        conv5_1 = slim.conv2d(pool4, 1024, [3,3], scope='conv5/conv5_1')
        conv5_2 = slim.conv2d(conv5_1, 1024, [3,3], scope='conv5/conv5_2')

        ###################
        # upsampling path #
        ###################
        conv6_1 = slim.conv2d_transpose(conv5_2, 512, [2,2], stride=2, scope='conv6/transpose_conv6_1')
        merge_1 = tf.concat([conv6_1, conv4_2], axis=-1, name='merge1') 
        conv6_2 = slim.conv2d(merge_1, 512, [3,3], scope='conv6/conv6_2')
        conv6_3 = slim.conv2d(conv6_2, 512, [3,3], scope='conv6/conv6_3')

        conv7_1 = slim.conv2d_transpose(conv6_3, 256, [2,2], stride=2, scope = 'conv7/transpose_conv7_1')
        merge_2 = tf.concat([conv7_1, conv3_2], axis=-1, name='merge2')
        conv7_2 = slim.conv2d(merge_2, 256, [3,3], scope='conv7/conv7_2')
        conv7_3 = slim.conv2d(conv7_2, 256, [3,3], scope='conv7/conv7_3')

        conv8_1 = slim.conv2d_transpose(conv7_3, 128, [2,2], stride=2, scope = 'conv8/transpose_conv8_1')
        merge_3 = tf.concat([conv8_1, conv2_2], axis=-1, name='merge3') 
        conv8_2 = slim.conv2d(merge_3, 128, [3,3], scope='conv8/conv8_2')
        conv8_3 = slim.conv2d(conv8_2, 128, [3,3], scope='conv8/conv8_3')

        conv9_1 = slim.conv2d_transpose(conv8_3, 64, [2,2], stride=2, scope = 'conv9/transpose_conv9_1')
        merge_4 = tf.concat([conv9_1, conv1_2], axis=-1, name='merge4') 
        conv9_2 = slim.conv2d(merge_4, 64, [3,3], scope='conv9/conv9_2')
        conv9_3 = slim.conv2d(conv9_2, 64, [3,3], scope='conv9/conv9_3')

        ###############
        # outpput map #
        ###############
        output_map = slim.conv2d(conv9_3, num_classes, [1, 1], 
                                activation_fn=None, normalizer_fn=None, 
                                scope='output_layer')

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return output_map, end_points

def UnetV2(inputs,
         num_classes = 2,
         dropout_keep_prob = 0.8,
         is_training = True,
         is_dropout = True,
         is_batch_norm = True,
         scope='unetV2'):
    """ Modifided Unet 
    Inputs: inputs - input image batch
            is_training - boolean whether to train graph or validate/test
            dropout_keep_prob - probability that each element is kept
            scope - scope name for model
    Outputs: output_map - output logits
             end_points - output dic
    """
    with tf.variable_scope(scope, 'unetV2', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):

        ######################
        # downsampling  path #
        ######################
        conv1 = conv_residual_block(inputs, 64, is_training, is_batch_norm, 'conv1')
        conv1_4 = slim.conv2d(conv1, 64, [1,1], activation_fn=None,  normalizer_fn=None, scope='conv1/conv1_4')
        if is_batch_norm:
            conv1_4 = slim.batch_norm(conv1_4, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='conv1/batch_norm4')
        conv1_4 = tf.nn.elu(conv1_4)
        pool1 = slim.max_pool2d(conv1_4, [2, 2], scope='pool1')

        conv2 = conv_residual_block(pool1, 128, is_training, is_batch_norm, 'conv2')
        conv2_4 = slim.conv2d(conv2, 128, [1,1], activation_fn=None,  normalizer_fn=None, scope='conv2/conv2_4')
        if is_batch_norm:
            conv2_4 = slim.batch_norm(conv2_4, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='conv2/batch_norm4')
        conv2_4 = tf.nn.elu(conv2_4)
        pool2 = slim.max_pool2d(conv2_4, [2, 2], scope='pool2')

        conv3 = conv_residual_block(pool2, 256, is_training, is_batch_norm, 'conv3')
        conv3_4 = slim.conv2d(conv3, 256, [1,1], activation_fn=None,  normalizer_fn=None, scope='conv3/conv3_4')
        if is_batch_norm:
            conv3_4 = slim.batch_norm(conv3_4, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='conv3/batch_norm4')
        conv3_4 = tf.nn.elu(conv3_4)
        pool3 = slim.max_pool2d(conv3_4, [2, 2], scope='pool3')

        conv4 = conv_residual_block(pool3, 512, is_training, is_batch_norm, 'conv4')
        conv4_4 = slim.conv2d(conv4, 512, [1,1], activation_fn=None,  normalizer_fn=None, scope='conv4/conv4_4')
        if is_batch_norm:
            conv4_4 = slim.batch_norm(conv4_4, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='conv4/batch_norm4')
        conv4_4 = tf.nn.elu(conv4_4)
        pool4 = slim.max_pool2d(conv4_4, [2, 2], scope='pool4')


        ##############
        # bottleneck #
        ##############
        conv5 = conv_residual_block(pool4, 1024, is_training, is_batch_norm, 'conv5')

        ###################
        # upsampling path #
        ###################
        conv6_up = slim.conv2d_transpose(conv5, 512, [1,1], normalizer_fn=None, stride=2, scope='conv6/transpose_conv6')
        conv6_up += conv4_4
        conv6_up = tf.nn.elu(conv6_up)
        conv6 = conv_residual_block(conv6_up, 512, is_training, is_batch_norm, 'conv6')
 
        conv7_up = slim.conv2d_transpose(conv6, 256, [1,1], normalizer_fn=None, stride=2, scope='conv7/transpose_conv7')
        conv7_up += conv3_4
        conv7_up = tf.nn.elu(conv7_up)
        conv7 = conv_residual_block(conv7_up, 256, is_training, is_batch_norm, 'conv7')

        conv8_up = slim.conv2d_transpose(conv7, 128, [1,1],normalizer_fn=None,  stride=2, scope='conv8/transpose_conv8')
        conv8_up += conv2_4
        conv8_up = tf.nn.elu(conv8_up)
        conv8 = conv_residual_block(conv8_up, 128, is_training, is_batch_norm, 'conv8')

        conv9_up = slim.conv2d_transpose(conv8, 64, [1,1], normalizer_fn=None, stride=2, scope='conv9/transpose_conv9')
        conv9_up += conv1_4
        conv9_up = tf.nn.elu(conv9_up)
        conv9 = conv_residual_block(conv9_up, 64, is_training, is_batch_norm, 'conv9')

        ###############
        # outpput map #
        ###############
        output_map = slim.conv2d(conv9, num_classes, [1, 1], 
                                activation_fn=None, normalizer_fn=None, 
                                scope='output_layer')

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return output_map, end_points


###############################################################################
# Residual Symmetric U-Net architecture: https://arxiv.org/pdf/1706.00120.pdf #
###############################################################################
def Residual_Symmetric_Unet(): 
    """ Coming soon!"""
    pass 












