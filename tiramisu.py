from __future__ import division 
import tensorflow as tf 
slim = tf.contrib.slim 

####################################################################
# One hundred layer tiramisu: https://arxiv.org/pdf/1611.09326.pdf #
####################################################################

def tiramisu_arg_scope(weight_decay=1e-04):
    """Defines the Tiramisu arg scope.
    Input: weight_decay - The l2 regularization coefficient
    Output: arg_scope - argument scope of model
    """
    with slim.arg_scope([slim.conv2d],
                      padding='SAME',
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()) as arg_sc:
        return arg_sc


def layer(net, dropout_keep_prob, is_training, layer_name):
    """ tiramisu layer that has dropout 
        Inputs: net - input feature map
                dropout_keep_prob - probability that each element is kept
                is_training - boolean whether to train graph or validate/test
                layer_name - scope name for layer

        Output: net - a feature map after non lineratiry, convolution and dropout is applied
    """
    with tf.variable_scope(layer_name):
        net = slim.batch_norm(net, is_training=is_training, decay=0.997, 
                epsilon=1e-5, center=True,scale=True,scope='batch_norm')
        net = tf.nn.relu(net)
        # growth rate set to 16
        net = slim.conv2d(net, 16, [3,3], activation_fn=None,  normalizer_fn=None, scope='conv')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout')
        return net

def transition_down(net, shape, dropout_keep_prob, is_training, transition_down_name):
    """ tiramisu downsampling module
        Inputs: net - input feature map
                shape - feature map shape
                dropout_keep_prob - probability that each element is kept
                is_training - boolean whether to train graph or validate/test
                transition_down_name - scope name for module
        Output: net - a feature map after non lineratiry, convolution and dropout is applied
    """
    with tf.variable_scope(transition_down_name):
        net = slim.batch_norm(net, is_training=is_training, decay=0.997, 
                epsilon=1e-5, center=True,scale=True,scope='batch_norm')
        net = tf.nn.relu(net)
        net = slim.conv2d(net, shape, [1, 1], activation_fn=None,  normalizer_fn=None, scope='conv')
        net = slim.dropout(net,dropout_keep_prob, is_training=is_training, scope='dropout')
        net = slim.max_pool2d(net, [2, 2], scope='pool')
        return net

def transition_up(net, shape, transition_up_name):
    """ tiramisu upsampling 
        Inputs: net - input feature map
                shape - feature map shape
                layer_name - scope name for upsampling
        Output: net - a feature map after non lineratiry, convolution and dropout is applied
    """
    with tf.variable_scope(transition_up_name):
        net = slim.conv2d_transpose(net, shape, [3,3], stride=2, scope = 'transpose_conv')
        return net

def dense_block(prev, num_layers, dropout_keep_prob, is_training, block_name):
    """ tiramisu dense block module
        Inputs: prev - previous input layer
                num_layers - number of layers in dense block
                dropout_keep_prob - probability that each element is kept
                is_training - boolean whether to train graph or validate/test
                block_name - scope name for block
        Ouptut: out - concatentation of previous layers 
    """

    with tf.variable_scope(block_name):

        collect = []
        for i in range(num_layers):
            current = layer(prev, dropout_keep_prob, is_training, 'layer_%d' % (i + 1))
            collect.append(current)
            prev = tf.concat([prev, current], axis=-1, name='concat_%d' % (i + 1))

        out = tf.concat(collect, axis=-1, name='concat_end')

        return out

def Tiramisu_103(inputs,
         is_training=True,
         num_classes = 2,
         dropout_keep_prob=0.75,
         scope='tiramisu'):
    """Tiramisu 103 fully convolutional densenet 
    Inputs: inputs - input image batch
            is_training - boolean whether to train graph or validate/test
            dropout_keep_prob - probability that each element is kept
            scope - scope name for model
    Outputs: output_map - output logits
             end_points - output dic
    """

    with tf.variable_scope(scope, 'tiramisu', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):

        #####################
        # input convolution #
        #####################
        input_conv = slim.conv2d(inputs, 48, [3,3],activation_fn=None,scope='input_conv')

        #####################
        # downsampling path #
        #####################
        dense_block_1 = dense_block(input_conv, 4, dropout_keep_prob, is_training, 'dense_block_1')
        dense_block_1 = tf.concat([dense_block_1, input_conv], axis=-1, name='down_concat_1')
        transition_down_1 = transition_down(dense_block_1,  112, dropout_keep_prob, is_training, 'transition_down_1')

        dense_block_2 = dense_block(transition_down_1, 5,  dropout_keep_prob, is_training,'dense_block_2')
        dense_block_2 = tf.concat([dense_block_2, transition_down_1], axis=-1, name='down_concat_2')
        transition_down_2 = transition_down(dense_block_2,  192, dropout_keep_prob, is_training, 'transition_down_2')

        dense_block_3 = dense_block(transition_down_2, 7, dropout_keep_prob, is_training, 'dense_block_3')
        dense_block_3 = tf.concat([dense_block_3, transition_down_2], axis=-1, name='down_concat_3')
        transition_down_3 = transition_down(dense_block_3,  304, dropout_keep_prob, is_training, 'transition_down_3')

        dense_block_4 = dense_block(transition_down_3, 10,  dropout_keep_prob, is_training,'dense_block_4')
        dense_block_4 = tf.concat([dense_block_4, transition_down_3], axis=-1, name='down_concat_4')
        transition_down_4 = transition_down(dense_block_4,  464, dropout_keep_prob, is_training, 'transition_down_4')

        dense_block_5 = dense_block(transition_down_4, 12,  dropout_keep_prob, is_training, 'dense_block_5')
        dense_block_5 = tf.concat([dense_block_5, transition_down_4], axis=-1, name='down_concat_5')
        transition_down_5 = transition_down(dense_block_5,  656, dropout_keep_prob, is_training, 'transition_down_5')

        ##############
        # bottleneck #
        ##############
        dense_block_6 = dense_block(transition_down_5, 15, dropout_keep_prob, is_training, 'dense_block_6')

        ###################
        # upsampling path #
        ###################
        transition_up_1 = transition_up(dense_block_6, 240, 'transition_up_1') 
        transition_up_1 = tf.concat([transition_up_1, dense_block_5], axis=-1, name='up_concat_1')
        dense_block_7 = dense_block(transition_up_1, 12, dropout_keep_prob, is_training, 'dense_block_7')

        transition_up_2 = transition_up(dense_block_7, 192,'transition_up_2') 
        transition_up_2 = tf.concat([transition_up_2, dense_block_4], axis=-1, name='up_concat_2')
        dense_block_8 = dense_block(transition_up_2, 10,  dropout_keep_prob, is_training, 'dense_block_8')

        transition_up_3 = transition_up(dense_block_8, 160, 'transition_up_3') 
        transition_up_3 = tf.concat([transition_up_3, dense_block_3], axis=-1, name='up_concat_3')
        dense_block_9 = dense_block(transition_up_3, 7, dropout_keep_prob, is_training,'dense_block_9')

        transition_up_4 = transition_up(dense_block_9, 112, 'transition_up_4') 
        transition_up_4 = tf.concat([transition_up_4, dense_block_2], axis=-1, name='up_concat_4')
        dense_block_10 = dense_block(transition_up_4, 5, dropout_keep_prob, is_training, 'dense_block_10')

        transition_up_5 = transition_up(dense_block_10, 80,'transition_up_5') 
        transition_up_5 = tf.concat([transition_up_5, dense_block_1], axis=-1, name='up_concat_5')
        dense_block_11 = dense_block(transition_up_5, 4,  dropout_keep_prob, is_training, 'dense_block_11')

        ###############
        # outpput map #
        ###############
        output_map = slim.conv2d(dense_block_11, num_classes, [1, 1], activation_fn=None, 
                                                normalizer_fn=None, scope='output_layer')

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return output_map, end_points






        



