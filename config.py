import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sparse

class ConnectomicsConfig():
    def __init__(self, **kwargs):

        # directories for storing tfrecords, checkpoints etc.
        self.main_dir = '/media/data_cifs/andreas/connectomics/'
        self.checkpoint_path = os.path.join(self.main_dir, 'checkpoints')
        self.summary_path = os.path.join(self.main_dir, 'summaries')
        self.results_path = os.path.join(self.main_dir, 'results')
       
        self.train_fn =  os.path.join(self.main_dir, 'tfrecords/ISBI_train.tfrecords')
        self.val_fn =  os.path.join(self.main_dir, 'tfrecords/ISBI_val.tfrecords')
        self.test_fn =  os.path.join(self.main_dir, 'tfrecords/ISBI_test.tfrecords')

        # self.test_checkpoint = os.path.join(self.checkpoint_path ,'unetV2/unetV2_Berson_2018_04_28_21_26_39_11100.ckpt')

        self.test_checkpoint = os.path.join(self.checkpoint_path ,'fusionNet/fusionNet_Berson_2018_05_07_21_54_13_1300.ckpt')
        #self.test_checkpoint = os.path.join(self.checkpoint_path ,'unet/unet_Berson_2018_04_07_12_16_00_15300.ckpt')
        # self.test_checkpoint = os.path.join(self.checkpoint_path ,'tiramisu/tiramisu_Berson_2018_04_29_19_33_14_23300.ckpt')


        self.optimizer = "adam"
        self.momentum = 0.95 # if optimizer is nestrov
        self.initial_learning_rate = 3e-04
        self.use_decay = True
        self.use_class_weights = False
        self.decay_steps = 675 # number of steps before decaying the learning rate
        self.learning_rate_decay_factor = 0.5 
        self.train_batch_size = 1
        self.val_batch_size = 1
        self.num_batches_to_validate_over = 1 # number of batches to validate over 
        self.validate_every_num_steps = 50 # perform a validation step
        self.num_train_epochs = 100
        self.output_shape = 2 # output shape of the model if 2 we have binary classification 
        self.input_image_size = [512, 512] # size of the input tf record image

        # various options for altering input images during training and validation
        self.train_augmentations_dic = {
                                        'rand_flip_left_right':True,
                                        'rand_flip_top_bottom':True,
                                        'rand_rotate':True,
                                        'rand_crop':True
                                       }

        self.val_augmentations_dic = {
                                      'rand_flip_left_right':False,
                                      'rand_flip_top_bottom':False,
                                      'rand_rotate':False,
                                      'rand_crop':False
                                     }
    

    def adapted_rand(self, seg, gt):
        """Compute Adapted Rand error as defined as 1 - the maximal F-score of the Rand index. 
        Implementation of the metric is taken from the CREMI contest:
        https://github.com/cremi/cremi_python/blob/master/cremi/evaluation/rand.py
        Input: seg - segmented mask
               gt - ground truth mask
        Output: are - A number between 0 and 1, lower number means better error
        """
        segA = np.ravel(gt)
        segB = np.ravel(seg)
        n = segA.size

        n_labels_A = np.amax(segA) + 1
        n_labels_B = np.amax(segB) + 1

        ones_data = np.ones(n)

        p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

        a = p_ij[1:n_labels_A,:]
        b = p_ij[1:n_labels_A,1:n_labels_B]
        c = p_ij[1:n_labels_A,0].todense()
        d = b.multiply(b)

        a_i = np.array(a.sum(1))
        b_i = np.array(b.sum(0))

        sumA = np.sum(a_i * a_i)
        sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
        sumAB = np.sum(d) + (np.sum(c) / n)

        precision = sumAB / sumB
        recall = sumAB / sumA

        fScore = 2.0 * precision * recall / (precision + recall)
        are = 1.0 - fScore

        return are.astype(np.float32)

    def a_rand(self, seg, gt):
        """Compute Adapted Rand error as defined as 1 - the maximal F-score of 
        the Rand index. Using py_func inorder to work with tensors
        Input: seg - segmented mask
               gt - ground truth mask
        Output: are - A number between 0 and 1, lower number means better error
        """
        return tf.py_func(self.adapted_rand, [seg,gt], tf.float32)


    def get_checkpoint_filename(self, model_name, run_name):
        """ 
        Return filename for a checkpoint file. Ensure path exists
        Input: model_name - Name of the model
               run_name - Timestap of the training 
        Output: Full checkpoint filepath
        """
        pth = os.path.join(self.checkpoint_path, model_name)
        if not os.path.isdir(pth): os.makedirs(pth)
        return os.path.join(pth, run_name + '.ckpt')

    def get_summaries_path(self, model_name, run_name):
        """ 
        Return filename for a summaries file. Ensure path exists
        Input: model_name - Name of the model
               run_name - Timestap of the training 
        Output: Full summaries filepath
        """
        pth = os.path.join(self.summary_path, model_name)
        if not os.path.isdir(pth): os.makedirs(pth)
        return os.path.join(pth, run_name)

    def get_results_path(self, model_name, run_name):
        """ 
        Return filename for a results file. Ensure path exists
        Input: model_name - Name of the model
               run_name - Timestap of the training 
        Output: Full results filepath
        """
        pth = os.path.join(self.results_path, model_name, run_name)
        if not os.path.isdir(pth): os.makedirs(pth)
        return pth