#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import argparse
import datetime
import numpy as np
import time
import unet
import tiramisu
from tensorflow.contrib import slim
from config import ConnectomicsConfig
from tf_record import preprocessing_Berson_with_mask, preprocessing_ISBI_with_mask, read_and_decode
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

def train(device):
    """
    Loads training and validations tf records and trains model and validates every number of fixed steps.
    Input: gpu device number 
    Output None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    config = ConnectomicsConfig() # loads configuration

    with tf.Graph().as_default():
        # load training data
        train_filename_queue = tf.train.string_input_producer(
        [config.train_fn], num_epochs=config.num_train_epochs)
        # load validation data
        val_filename_queue = tf.train.string_input_producer(
        [config.val_fn], num_epochs=config.num_train_epochs)

        # defining model names and setting output and summary directories
        model_train_name = 'unetV2'
        dt_stamp = time.strftime("Berson_%Y_%m_%d_%H_%M_%S")
        out_dir = config.get_results_path(model_train_name, dt_stamp)
        summary_dir = config.get_summaries_path(model_train_name, dt_stamp)
        print '-'*60
        print 'Training model: {0}'.format(dt_stamp)
        print '-'*60

        train_images, train_masks = read_and_decode(filename_queue = train_filename_queue,
                                                         img_dims = config.input_image_size,
                                                         size_of_batch = config.train_batch_size,
                                                          augmentations_dic = config.train_augmentations_dic,
                                                         num_of_threads = 2,
                                                         shuffle = True)

        val_images, val_masks  = read_and_decode(filename_queue = val_filename_queue,
                                                     img_dims = config.input_image_size,
                                                     size_of_batch =  config.val_batch_size,
                                                     augmentations_dic = config.val_augmentations_dic,
                                                     num_of_threads = 1,
                                                     shuffle = False)

        # summaries to use with tensorboard check https://www.tensorflow.org/get_started/summaries_and_tensorboard
        tf.summary.image('train images', train_images, max_outputs=1)
        tf.summary.image('train masks', train_masks, max_outputs=1)

        tf.summary.image('validation images', val_images, max_outputs=1)
        tf.summary.image('validation masks', val_masks, max_outputs=1)

        # creating step op that counts the number of training steps
        step = get_or_create_global_step()
        step_op = tf.assign(step, step+1)

        with tf.variable_scope('unetV2') as unet_scope:
            with tf.name_scope('train') as train_scope:

                train_processed_images, train_processed_masks = preprocessing_Berson_with_mask(train_images, train_masks)
                with slim.arg_scope(unet.unet_arg_scope()):
                    train_logits, _ = unet.UnetV2(train_processed_images,
                                                is_training=True,
                                                num_classes = config.output_shape,
                                                scope=unet_scope)
                    train_prob = tf.nn.softmax(train_logits)
                    train_scores = tf.argmax(train_prob, axis=3)
                    train_a_rand = config.a_rand(train_scores, train_processed_masks)

                tf.summary.scalar("Train ARAND", train_a_rand)
                
                flatten_train_processed_masks = tf.reshape(train_processed_masks, [-1])
                flatten_train_logits = tf.reshape(train_logits, [-1, config.output_shape])

                one_hot_lables = tf.one_hot(flatten_train_processed_masks, config.output_shape, axis=-1)

                # class_weights = tf.constant(np.load('class_weights_Berson.npy'))
                # weight_map = tf.multiply(one_hot_lables, class_weights)
                # weight_map = tf.reduce_sum(weight_map, axis=1)

                batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_lables, logits = flatten_train_logits)
                
                # weighted_loss = tf.multiply(batch_loss, weight_map)

                if config.use_class_weights:
                    batch_loss = weighted_loss

                loss = tf.reduce_mean(batch_loss)
                tf.summary.scalar("loss", loss)

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
                val_processed_images, val_processed_masks = preprocessing_Berson_with_mask(val_images, val_masks)

                with slim.arg_scope(unet.unet_arg_scope()):
                    val_logits, _ = unet.UnetV2(val_processed_images,
                                                is_training=False,
                                                num_classes = config.output_shape,
                                                scope=unet_scope)

                    val_prob = tf.nn.softmax(val_logits)
                    val_scores = tf.argmax(val_prob, axis=3)
                    val_a_rand = config.a_rand(val_scores, val_processed_masks)

                tf.summary.scalar("Validation ARAND", val_a_rand)

        saver = tf.train.Saver(slim.get_model_variables(), max_to_keep=100)

        summary_op = tf.summary.merge_all()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            sess.run(tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()))
            summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            np.save(os.path.join(out_dir, 'training_config_file'), config)
            val_a_rand_min, losses = float('inf'), []

            try:

                while not coord.should_stop():

                    start_time = time.time()
                    step_count, loss_value, train_a_rand_value, lr_value, _ = sess.run([step_op, loss, train_a_rand, lr, train_op])
                    losses.append(loss_value)
                    duration = time.time() - start_time
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    step_count = step_count - 1 

                    if step_count % config.validate_every_num_steps == 0:
                        # Summaries
                        it_a_arand = np.asarray([])
                        for num_vals in range(config.num_batches_to_validate_over):
                            # Validation accuracy as the average of n batches
                            it_a_arand = np.append(
                                it_a_arand, sess.run(val_a_rand))
                        
                        val_a_rand_total = it_a_arand.mean()

                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step_count)

                        # Training status and validation accuracy
                        msg = '{0}: step {1}, loss = {2:.2f} ({3:.2f} examples/sec; '\
                            + '{4:.2f} sec/batch] | Training ARAND = {5:.6f}) '\
                            +  '| Validation ARAND = {6:.6f} | logdir = {7}'
                        print msg.format(
                              datetime.datetime.now(), step_count, loss_value,
                              (config.train_batch_size / duration), float(duration),
                              train_a_rand_value, val_a_rand_total, summary_dir)
                        print "learning rate: ", lr_value

                        # Save the model checkpoint if it's the best yet
                        if val_a_rand_total <= val_a_rand_min:
                            file_name = 'unetV2_{0}_{1}'.format(dt_stamp, step_count)
                            saver.save(
                                sess,
                                config.get_checkpoint_filename(model_train_name, file_name))
                        # Store the new max validation accuracy
                        val_a_rand_max = val_a_rand_total
                    
                    else:
                        # Training status
                        msg = '{0}: step {1}, loss = {2:.2f} ({3:.2f} examples/sec; '\
                            + '{4:.2f} sec/batch | Training ARAND =  {5:.6f})'
                        print msg.format(datetime.datetime.now(), step_count, loss_value,
                              (config.train_batch_size / duration),
                              float(duration),train_a_rand_value)
                    # End iteration

            except tf.errors.OutOfRangeError:
                print 'Done training for {0} epochs, {1} steps.'.format(config.num_train_epochs, step_count)
            finally:
                coord.request_stop()
                np.save(os.path.join(out_dir, 'training_loss'), losses)
            coord.join(threads)
            sess.close()
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    args = parser.parse_args()
    train(args.device) # select gpu to train model on