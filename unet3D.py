from __future__ import division 
import tensorflow as tf 
slim = tf.contrib.slim 



def conv3d(net, D, H, W, output_dim, is_training, batch_norm, acitvation, layer_name):
    with tf.variable_scope(layer_name):

        w = tf.get_variable('w', [D, H, W, net.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv3d(net, w, strides=[1,1,1,1,1], padding='SAME')
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0)) 
        conv = tf.nn.bias_add(conv, b)
        if batch_norm:
            bn = tf.contrib.layers.batch_norm(conv, is_training=is_training, scope='bn',
                                     decay=0.997, epsilon=1e-5, center=True, scale=True)
        if acitvation == 'elu':
            r = tf.nn.elu(bn)
        elif acitvation == 'relu':
            r = tf.nn.relu(bn)
        elif acitvation == 'None':
            return bn 
        else:
            raise Exception('activation fuction not available')
        return r

def transpose3d(net, D, H, W, output_shape, is_training, batch_norm, acitvation, layer_name):
    with tf.variable_scope(layer_name):
        output_dim = output_shape[-1]
        w = tf.get_variable('w', [D, H, W, output_dim, net.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        trans = tf.nn.conv3d_transpose(net, w, output_shape, strides=[1, 1, 2, 2, 1], padding='SAME')
        b = tf.get_variable('b', [output_shape], initializer=tf.constant_initializer(0.0))
        trans = tf.nn.bias_add(trans, b)
        if batch_norm:
            bn = tf.contrib.layers.batch_norm(trans, is_training=is_training, scope='bn',
                                     decay=0.997, epsilon=1e-5, center=True, scale=True)
        if acitvation == 'elu':
            r = tf.nn.elu(bn)
        elif acitvation == 'relu':
            r = tf.nn.relu(bn)
        elif acitvation == 'None':
            return bn 
        else:
            raise Exception('activation fuction not available')
        return r

def residualblock3D(net, output_dim, is_training, layer_name):
     with tf.variable_scope(layer_name):
        net = conv3d(net, 1, 3, 3, output_dim, is_training, True, 'elu', 'conv%d_1' % int(layer_name[-1]))
        skip = net
        net = conv3d(net, 3, 3, 3, output_dim, is_training, True, 'elu', 'conv%d_2' % int(layer_name[-1]))
        net = conv3d(net, 3, 3, 3, output_dim, is_training, False, 'None', 'conv%d_3' % int(layer_name[-1]))
        net = net + skip
        net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='bn',
                                 decay=0.997, epsilon=1e-5, center=True, scale=True)
        return tf.nn.elu(net)

def up_skip(conv, skip):
    net = (conv + skip)
    net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='bn',
                                 decay=0.997, epsilon=1e-5, center=True, scale=True)
    return tf.nn.elu(net)

def Unet3D(inputs,
         num_classes = 2,
         is_training = True,
         is_batch_norm = False,
         scope='unet3D'):
    with tf.variable_scope(scope, 'unet3D', [inputs]) as sc:

        ######################
        # downsampling  path #
        ######################
        conv1_1 = conv3d(inputs, 3, 3, 3, 32, is_training, True, 'relu', 'conv1/conv1_1')
        conv1_2 = conv3d(conv1_1,3, 3, 3, 64, is_training, True, 'relu', 'conv1/conv1_2')
        pool1 = tf.nn.max_pool3d(conv1_2,ksize=[1, 2, 2, 2, 1],
                                        strides=[1, 2, 2, 2, 1],
                                        padding='SAME')

        conv2_1 = conv3d(pool1,3, 3, 3, 64, is_training, True, 'relu', 'conv2/conv2_1')
        conv2_2 = conv3d(conv2_1,3, 3, 3, 128, is_training, True, 'relu', 'conv2/conv2_2')
        pool2 = tf.nn.max_pool3d(conv2_2,ksize=[1, 2, 2, 2, 1],
                                        strides=[1, 2, 2, 2, 1],
                                        padding='SAME')

        conv3_1 = conv3d(pool2,3, 3, 3, 128, is_training, True, 'relu', 'conv3/conv3_1')
        conv3_2 = conv3d(conv3_1,3, 3, 3, 256, is_training, True, 'relu', 'conv3/conv3_2')
        pool3 = tf.nn.max_pool3d(conv3_2,ksize=[1, 2, 2, 2, 1],
                                        strides=[1, 2, 2, 2, 1],
                                        padding='SAME')

        ##############
        # bottleneck #
        ##############
        conv4_1 = conv3d(pool3,3, 3, 3, 256, is_training, True, 'relu', 'conv4/conv4_1')
        conv4_2 = conv3d(conv4_1,3, 3, 3, 512, is_training, True, 'relu', 'conv4/conv4_2')

        ###################
        # upsampling path #
        ###################
        shape = conv4_2.get_shape().as_list()
        transpose_output_shape = [tf.shape(conv4_2)[0], shape[1] * 2, shape[2] * 2,
                                   shape[3] * 2, 512]
        conv5_1 = transpose3d(conv4_2, 2, 2, 2, transpose_output_shape, is_training, True, 'relu', 
                                                                    'conv5/transpose_conv5_1')
        merge_1 = tf.concat([conv5_1, conv3_2], axis=-1, name='merge1') 
        conv5_2 = conv3d(merge_1,3, 3, 3, 256, is_training, True, 'relu', 'conv5/conv5_2')
        conv5_3 = conv3d(conv5_2,3, 3, 3, 256, is_training, True, 'relu', 'conv5/conv5_3')

        shape = conv5_3.get_shape().as_list()
        transpose_output_shape = [tf.shape(conv5_3)[0], shape[1] * 2, shape[2] * 2,
                                   shape[3] * 2, 256]
        conv6_1 = transpose3d(conv5_3, 2, 2, 2, transpose_output_shape, is_training, True, 'relu', 
                                                                    'conv6/transpose_conv6_1')
        merge_2 = tf.concat([conv6_1, conv2_2], axis=-1, name='merge2') 
        conv6_2 = conv3d(merge_2,3, 3, 3, 128, is_training, True, 'relu', 'conv6/conv6_2')
        conv6_3 = conv3d(conv6_2,3, 3, 3, 128, is_training, True, 'relu', 'conv6/conv6_3')

        shape = conv6_3.get_shape().as_list()
        transpose_output_shape = [tf.shape(conv6_3)[0], shape[1] * 2, shape[2] * 2,
                                   shape[3] * 2, 128]
        conv7_1 = transpose3d(conv6_3, 2, 2, 2, transpose_output_shape, is_training, True, 'relu', 
                                                                    'conv7/transpose_conv7_1')
        merge_3 = tf.concat([conv7_1, conv1_2], axis=-1, name='merge3') 
        conv7_2 = conv3d(merge_3,3, 3, 3, 64, is_training, True, 'relu', 'conv7/conv7_2')
        conv7_3 = conv3d(conv7_2,3, 3, 3, 64, is_training, True, 'relu', 'conv7/conv7_3')

        with tf.variable_scope('logits') as scope:
            w = tf.get_variable('w', [1, 1, 1, conv7_3.get_shape()[-1], num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.nn.conv3d(conv7_3, w, strides=[1, 1, 1, 1, 1], padding='SAME')
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(logits, b)

        return logits

def ResidualUnet3D(inputs,
         num_classes = 1,
         is_training = True,
         is_batch_norm = False,
         scope='resUnet3D'):
    with tf.variable_scope(scope, 'residualUnet3D', [inputs]) as sc:

        ######################
        # downsampling  path #
        ######################
        input_conv = conv3d(inputs, 1, 5, 5, 28, is_training, True, 'elu', 'input_conv')
        conv1 = residualblock3D(input_conv, 28, is_training,'conv1')
        pool1 = tf.nn.max_pool3d(conv1,ksize=[1, 1, 2, 2, 1],
                                        strides=[1, 1, 2, 2, 1],
                                        padding='SAME')

        conv2 = residualblock3D(pool1, 36, is_training, 'conv2')
        pool2 = tf.nn.max_pool3d(conv2,ksize=[1, 1, 2, 2, 1],
                                        strides=[1, 1, 2, 2, 1],
                                        padding='SAME')

        conv3 = residualblock3D(pool2, 48, is_training, 'conv3')
        pool3 = tf.nn.max_pool3d(conv3,ksize=[1, 1, 2, 2, 1],
                                        strides=[1, 1, 2, 2, 1],
                                        padding='SAME')

        conv4 = residualblock3D(pool3, 64, is_training,'conv4')
        pool4 = tf.nn.max_pool3d(conv4,ksize=[1, 1, 2, 2, 1],
                                        strides=[1, 1, 2, 2, 1],
                                        padding='SAME')

        ##############
        # bottleneck #
        ##############
        conv5 = residualblock3D(pool4, 80, is_training, 'conv5')

        ###################
        # upsampling path #
        ###################
        shape = conv5.get_shape().as_list()
        transpose_output_shape = [tf.shape(conv5)[0], shape[1], shape[2] * 2,
                                   shape[3] * 2, 80]
        conv6 = transpose3d(conv5, 1, 2, 2, transpose_output_shape, is_training, 'None', 
                                                                    'conv6/transpose_conv')
        conv6 = up_skip(conv6, conv4)
        conv6 = residualblock3D(conv6, 64, is_training,'conv6')

        shape = conv6.get_shape().as_list()
        transpose_output_shape = [tf.shape(conv6)[0], shape[1], shape[2] * 2,
                                   shape[3] * 2, 64]
        conv7 = transpose3d(conv6, 1, 2, 2, transpose_output_shape, is_training, True, 'None', 
                                                                    'conv7/transpose_conv')
        conv7 = up_skip(conv7, conv3)
        conv7 = residualblock3D(conv7, 48, is_training,'conv7')

        shape = conv7.get_shape().as_list()
        transpose_output_shape = [tf.shape(conv7)[0], shape[1], shape[2] * 2,
                                   shape[3] * 2, 48]
        conv8 = transpose3d(conv7, 1, 2, 2, transpose_output_shape, is_training, True, 'None', 
                                                                    'conv8/transpose_conv')
        conv8 = up_skip(conv8, conv2)
        conv8 = residualblock3D(conv8, 36, is_training, 'conv8')

        shape = conv8.get_shape().as_list()
        transpose_output_shape = [tf.shape(conv8)[0], shape[1], shape[2] * 2,
                                   shape[3] * 2, 36]
        conv9 = transpose3d(conv9, 1, 2, 2, transpose_output_shape, is_training, True, 'None', 
                                                                    'conv9/transpose_conv')
        conv9 = up_skip(conv9, conv1)
        conv9 = residualblock3D(conv9, 28, is_training, 'conv9')
        output_conv = conv3d(conv9, 1, 5, 5, 28, is_training, True, 'elu', 'output_conv')

        with tf.variable_scope('logits') as scope:
            w = tf.get_variable('w', [1, 1, 1, output_conv.get_shape()[-1], num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.nn.conv3d(output_conv, w, strides=[1, 1, 1, 1, 1], padding='SAME')
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(logits, b)

        return logits


        

        


        