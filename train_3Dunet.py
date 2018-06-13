#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import argparse
import datetime
import numpy as np
import time
import unet3D
from data_utils import *
from config import ConnectomicsConfig
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

def train(device):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    config = ConnectomicsConfig() # loads configuration

    volume, affinity = load_images_and_labels(config.data_dir)

    np_train_volume = volume[:80,:,:]
    np_train_aff = affinity[:80,:,:]

    np_val_volume = volume[80:,:,:]
    np_val_aff = affinity[80:,:,:]

    np_train_volume = np.expand_dims(np_train_volume, -1)
    np_train_aff = np.expand_dims(np_train_aff, -1)

    np_val_volume = np.expand_dims(np_val_volume, -1)
    np_val_aff = np.expand_dims(np_val_aff, -1)

    with tf.Graph().as_default():

        # defining model names and setting output and summary directories
        model_train_name = 'Unet3D'
        dt_stamp = time.strftime("ISBI3D_%Y_%m_%d_%H_%M_%S")
        out_dir = config.get_results_path(model_train_name, dt_stamp)
        summary_dir = config.get_summaries_path(model_train_name, dt_stamp)
        print '-'*60
        print 'Training model: {0}'.format(dt_stamp)
        print '-'*60

        train_volume = tf.placeholder(tf.float32, shape=[None,16,128,128,1], name='train_volume')
        train_aff = tf.placeholder(tf.int32, shape=[None,16,128,128,1], name='train_aff')

        val_volume = tf.placeholder(tf.float32, shape=[None,16,128,128,1], name='val_volume')
        val_aff = tf.placeholder(tf.int32, shape=[None,16,128,128,1], name='val_aff')
        # # summaries to use with tensorboard check https://www.tensorflow.org/get_started/summaries_and_tensorboard
        # tf.summary.image('train images', train_images, max_outputs=1)
        # tf.summary.image('train masks', train_masks, max_outputs=1)

        # tf.summary.image('validation images', val_images, max_outputs=1)
        # tf.summary.image('validation masks', val_masks, max_outputs=1)

        # creating step op that counts the number of training steps
        step = get_or_create_global_step()
        step_op = tf.assign(step, step+1)

        with tf.variable_scope('Unet3D') as unet_scope:
            with tf.name_scope('train') as train_scope:

                train_logits = unet3D.Unet3D(train_volume,
                                            is_training=True,
                                            num_classes = config.output_shape,
                                            scope=unet_scope)
                    
                train_prob = tf.nn.softmax(train_logits)
                train_scores = tf.argmax(train_prob, axis=4)
                train_a_rand = config.a_rand(train_scores, train_aff)

                # tf.summary.scalar("Train ARAND", train_a_rand)
                
                flatten_train_aff = tf.reshape(train_aff, [-1])
                flatten_train_logits = tf.reshape(train_logits, [-1, config.output_shape])

                one_hot_lables = tf.one_hot(flatten_train_aff, config.output_shape, axis=-1)

                # class_weights = tf.constant(np.load('class_weights_Berson.npy'))
                # weight_map = tf.multiply(one_hot_lables, class_weights)
                # weight_map = tf.reduce_sum(weight_map, axis=1)

                batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_lables, logits = flatten_train_logits)
                
                # weighted_loss = tf.multiply(batch_loss, weight_map)

                if config.use_class_weights:
                    batch_loss = weighted_loss

                loss = tf.reduce_mean(batch_loss)
                # tf.summary.scalar("loss", loss)

                if config.use_decay:
                    lr = tf.train.exponential_decay(
                            learning_rate = config.initial_learning_rate,
                            global_step = step_op,
                            decay_steps = config.decay_steps,
                            decay_rate = config.learning_rate_decay_factor,
                            staircase = True) # if staircase is True decay the learning rate at discrete intervals
                else:
                    lr = tf.constant(config.initial_learning_rate)


                if config.optimizer == "adam":
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # used to update batch norm params. see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
                    with tf.control_dependencies(update_ops):
                        train_op =  tf.train.AdamOptimizer(lr).minimize(loss)
                elif config.optimizer == "sgd":
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        train_op =  tf.train.GradientDescentOptimizer(lr).minimize(loss)
                elif config.optimizer == "nestrov":
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        train_op =  tf.train.MomentumOptimizer(lr, config.momentum, use_nesterov=True).minimize(loss)
                else:
                    raise Exception("Not known optimizer! options are adam, sgd or nestrov")
                   
            unet_scope.reuse_variables() # training variables are reused in validation graph 

            with tf.name_scope('val') as val_scope:

                val_logits = unet3D.Unet3D(val_volume,
                                        is_training=False,
                                        num_classes = config.output_shape,
                                        scope=unet_scope)

                val_prob = tf.nn.softmax(val_logits)
                val_scores = tf.argmax(val_prob, axis=4)
                val_a_rand = config.a_rand(val_scores, val_aff)

                # tf.summary.scalar("Validation ARAND", val_a_rand)

        saver = tf.train.Saver(max_to_keep=100)

        # summary_op = tf.summary.merge_all()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            sess.run(tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()))
            # summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

            np.save(os.path.join(out_dir, 'training_config_file'), config)
            val_a_rand_min, losses = float('inf'), []

            for _ in range(config.iters):

                crop_train_volume, indiciies = random_crop_volume(np_train_volume,config.crop_size)
                crop_train_aff = crop_volume(np_train_aff,config.crop_size, indiciies)

                start_time = time.time()
                step_count, loss_value, train_a_rand_value, lr_value, _ = sess.run([step_op, loss, train_a_rand, lr, train_op],feed_dict={train_volume: crop_train_volume,
                                                                                                                              train_aff: crop_train_aff})
                losses.append(loss_value)
                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                step_count -= 1 

                if step_count % config.validate_every_num_steps == 0:
                    # Summaries
                    it_a_rand = np.asarray([])
                    for num_vals in range(config.num_batches_to_validate_over):

                        crop_val_volume, indiciies = random_crop_volume(np_val_volume,config.crop_size)
                        crop_val_aff = crop_volume(np_val_aff,config.crop_size, indiciies)
                        # Validation accuracy as the average of n batches
                        it_a_rand = np.append(
                            it_a_rand, sess.run(val_a_rand,feed_dict={val_volume: crop_val_volume,
                                                                      val_aff: crop_val_aff}))
                    
                    val_a_rand_total = it_a_rand.mean()

                    # summary_str = sess.run(summary_op)
                    # summary_writer.add_summary(summary_str, step_count)

                    # Training status and validation accuracy
                    msg = '{0}: step {1}, loss = {2:.2f} ({3:.2f} examples/sec; '\
                        + ' | Training ARAND = {4:.6f}) '\
                        +  '| Validation ARAND = {5:.6f} | logdir = {6}'
                    print msg.format(
                          datetime.datetime.now(), step_count, loss_value,
                           float(duration),
                          train_a_rand_value, val_a_rand_total, summary_dir)
                    print "learning rate: ", lr_value

                    # Save the model checkpoint if it's the best yet
                    if val_a_rand_total <= val_a_rand_min:
                        file_name = 'Unet3D_{0}_{1}'.format(dt_stamp, step_count)
                        saver.save(
                            sess,
                            config.get_checkpoint_filename(model_train_name, file_name))
                    
                        val_a_rand_min = val_a_rand_total
                
                else:
                    # Training status
                    msg = '{0}: step {1}, loss = {2:.2f} ({3:.2f} examples/sec; '\
                        + '| Training ARAND =  {4:.6f})'
                    print msg.format(datetime.datetime.now(), step_count, loss_value,
                          float(duration),train_a_rand_value)
                # End iteration

        print 'Done training for {0} epochs, {1} steps.'.format(config.num_train_epochs, step_count)
        np.save(os.path.join(out_dir, 'training_loss'), losses)
        sess.close()
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    args = parser.parse_args()
    train(args.device) # select gpu to train model on