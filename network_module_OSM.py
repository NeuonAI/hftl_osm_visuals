# Neuon AI - PlantCLEF 2021

"""
Network module for OSM
current architecture: inception v4
- change network architecture accordingly
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import sys
sys.path.append("PATH_TO_SLIM") # path to /models/research/slim
import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim
import numpy as np
import cv2
from nets.inception_v4 import inception_v4
from nets import inception_utils
from PIL import Image

class network_module(object):
    def __init__(self,
                 batch,
                 iterbatch,
                 numclasses,
                 image_dir_parent_train,
                 image_dir_parent_test,
                 train_file,
                 test_file,
                 input_size,
                 checkpoint_model,
                 learning_rate,
                 save_dir,
                 tensorboard_dir,
                 max_iter,
                 val_freq,
                 val_iter):
        
        self.batch = batch
        self.iterbatch = iterbatch
        self.image_dir_parent_train = image_dir_parent_train
        self.image_dir_parent_test = image_dir_parent_test
        self.train_file = train_file
        self.test_file = test_file
        self.input_size = input_size
        self.numclasses = numclasses
        self.checkpoint_model = checkpoint_model
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.tensorboard_dir = tensorboard_dir
        self.max_iter = max_iter
        self.val_freq = val_freq
        self.val_iter = val_iter
        
        
        self.train_database = database_module(
                image_source_dir = self.image_dir_parent_train,
                database_file = self.train_file,
                batch = self.batch,
                input_size = self.input_size,
                numclasses = self.numclasses,
                shuffle = True)

        self.test_database = database_module(
                image_source_dir = self.image_dir_parent_test,
                database_file = self.test_file,
                batch = self.batch,
                input_size = self.input_size,
                numclasses = self.numclasses,
                shuffle = True)
        

        # ----- Tensors ------ #         
        x = tf.placeholder(tf.float32,(self.batch,) + self.input_size)
        y1 = tf.placeholder(tf.int32,(self.batch,))
        y2 = tf.placeholder(tf.int32,(self.batch,))
        y3 = tf.placeholder(tf.int32,(self.batch,))
        y_onehot1 = tf.one_hot(y1,self.numclasses)
        self.is_training = tf.placeholder(tf.bool)

        # ----- Image pre-processing methods ----- # 
        train_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=True)
        
        def data_in_train():
            return tf.map_fn(fn = train_preproc,elems = x,dtype=np.float32)
        
        test_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=False)        
        
        def data_in_test():
            return tf.map_fn(fn = test_preproc,elems = x,dtype=np.float32)
        
        data_in = tf.cond(
                self.is_training,
                true_fn = data_in_train,
                false_fn = data_in_test
                )

        # ----- Network construction ----- #        
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits,endpoints = inception_v4(data_in,
                                            num_classes=self.numclasses,
                                            is_training=self.is_training)
            
            
            feat = endpoints['PreLogitsFlatten']     
            feat_500 = tf.contrib.layers.fully_connected(
                            inputs=feat,
                            num_outputs=500,
                            activation_fn=None,
                            normalizer_fn=None,
                            trainable=True,
                            scope='feat_500'                            
                    )
            logits_500 = slim.fully_connected(feat_500,997,activation_fn=None,
                                        scope='Species_500')
             
            
        # ----- Network losses ----- #
        with tf.name_scope("cross_entropy"): 
            with tf.name_scope("auxloss"):
                self.auxloss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=endpoints['AuxLogits'], labels=y_onehot1))
            
            with tf.name_scope("logits_loss_500"):
                self.loss_500 = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits_500, labels=y_onehot1))
            
            with tf.name_scope("logits_loss"):
                self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=logits, labels=y_onehot1))                



            with tf.name_scope("L2_reg_loss"):
                self.regularization_loss = 0.00004 * tf.add_n(tf.losses.get_regularization_losses(scope='InceptionV4'))
            with tf.name_scope("total_loss"):
                self.totalloss = self.loss_500 + self.loss + self.auxloss + self.regularization_loss

            
        # ----- Accuracy ----- #
        with tf.name_scope("accuracy"):
            with tf.name_scope('accuracy_500'):
                prediction0 = tf.argmax(logits_500,1)
                match = tf.equal(prediction0,tf.argmax(y_onehot1,1))
                self.accuracy_500 = tf.reduce_mean(tf.cast(match,tf.float32))             
            with tf.name_scope('accuracy_species'):
                prediction = tf.argmax(logits,1)
                match = tf.equal(prediction,tf.argmax(y_onehot1,1))
                self.accuracy = tf.reduce_mean(tf.cast(match,tf.float32))         
            

        # ----- Get all variables ----- #
        self.var_list = [v for v in tf.trainable_variables()]

        self.var_list_train = self.var_list
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        
        # ----- Restore model ----- #
        self.variables_to_restore = slim.get_variables_to_restore()
        restore_fn = slim.assign_from_checkpoint_fn(
            self.checkpoint_model, self.variables_to_restore[0:-20])  
        

        # ----- Training scope ----- # 
        with tf.name_scope("train"):

            loss_accumulator = tf.Variable(0.0, trainable=False)
            acc_accumulator_500 = tf.Variable(0.0, trainable=False)
            acc_accumulator = tf.Variable(0.0, trainable=False)

            
            self.collect_loss = loss_accumulator.assign_add(self.totalloss)
            self.collect_acc_500 = acc_accumulator_500.assign_add(self.accuracy_500)              
            self.collect_acc = acc_accumulator.assign_add(self.accuracy)          

                        
            self.average_loss = tf.cond(self.is_training,
                                        lambda: loss_accumulator / self.iterbatch,
                                        lambda: loss_accumulator / self.val_iter)
            self.average_acc_500 = tf.cond(self.is_training,
                                       lambda: acc_accumulator_500 / self.iterbatch,
                                       lambda: acc_accumulator_500 / self.val_iter)
            self.average_acc = tf.cond(self.is_training,
                                       lambda: acc_accumulator / self.iterbatch,
                                       lambda: acc_accumulator / self.val_iter)


            
            self.zero_op_loss = tf.assign(loss_accumulator,0.0)
            self.zero_op_acc_500 = tf.assign(acc_accumulator_500,0.0)
            self.zero_op_acc = tf.assign(acc_accumulator,0.0)


            self.accum_train = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_train]                                        
            self.zero_ops_train = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train]
            
            
            # ----- Set up optimizer / Compute gradients ----- #
            with tf.control_dependencies(self.update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                gradient = optimizer.compute_gradients(self.totalloss,self.var_list_train)
                
                gradient_only = [gc[0] for gc in gradient]
                gradient_only,_ = tf.clip_by_global_norm(gradient_only,1.25)
                
                self.accum_train_ops = [self.accum_train[i].assign_add(gc) for i,gc in enumerate(gradient_only)]
            self.train_step = optimizer.apply_gradients(
                    [(self.accum_train[i], gc[1]) for i, gc in enumerate(gradient)])

            
        # ----- Global variables ----- #
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        
        # ----- Create saver ----- #
        saver = tf.train.Saver(var_list=var_list, max_to_keep=0)

        tf.summary.scalar('loss',self.average_loss)
        tf.summary.scalar('accuracy_500',self.average_acc_500) 
        tf.summary.scalar('accuracy',self.average_acc) 


        self.merged = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'accuracy'),
                                        tf.get_collection(tf.GraphKeys.SUMMARIES,'loss')])
        
        # ----- Tensorboard writer--- #    
        tensorboar_dir = self.tensorboard_dir + '/run1_OSM_tensorboard'
        writer_train = tf.summary.FileWriter(tensorboar_dir+'/train')
        writer_test = tf.summary.FileWriter(tensorboar_dir+'/test')


        # ----- Create session ----- #
        val_best = 0.0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer_train.add_graph(sess.graph)
            writer_test.add_graph(sess.graph)
            
            restore_fn(sess)

            
            for i in range(self.max_iter+1):
                try:
                    sess.run(self.zero_ops_train)
                    sess.run([self.zero_op_acc,self.zero_op_acc_500,self.zero_op_loss])
                    
                    # ----- Validation ----- #
                    if i % self.val_freq == 0:                        
                        print('Start:%f'%sess.run(loss_accumulator))
                        for j in range(self.val_iter):
                            img,lbl1,lbl2,lbl3 = self.test_database.read_batch()
                            sess.run(
                                        [self.collect_loss,self.collect_acc_500,self.collect_acc],
                                        feed_dict = {x : img,
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     y3 : lbl3,
                                                     self.is_training : False
                                        }                                  
                                    )
                            print('[%i]:%f'%(j,sess.run(loss_accumulator)))
                        print('End:%f'%sess.run(loss_accumulator))  
                        s,self.netLoss,self.netAccuracy500,self.netAccuracy = sess.run(
                                [self.merged,self.average_loss,self.average_acc_500,self.average_acc],
                                    feed_dict = {
                                            self.is_training : False
                                    }                            
                                ) 
                        writer_test.add_summary(s, i)
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy:%f'%(self.train_database.epoch,i,self.netLoss,self.netAccuracy))
                        print('[Valid] Epoch:%i Iter:%i Loss:%f, Accuracy 500 feat:%f'%(self.train_database.epoch,i,self.netLoss,self.netAccuracy500))


                        sess.run([self.zero_op_acc,self.zero_op_acc_500,self.zero_op_loss])
                        
                        if self.netAccuracy > val_best:
                            val_best = self.netAccuracy
                            saver.save(sess, os.path.join(self.save_dir,'best.ckpt'))
                            print('Model saved')




                    # ----- Train ----- #
                    for j in range(self.iterbatch):
                        img,lbl1,lbl2,lbl3 = self.train_database.read_batch()
                        
    
                        sess.run(
                                    [self.collect_loss,self.collect_acc_500,self.collect_acc,self.accum_train_ops],
                                    feed_dict = {x : img,
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     y3 : lbl3,
                                                 self.is_training : True
                                    }                                
                                )
                        
                    s,self.netLoss,self.netAccuracy500,self.netAccuracy = sess.run(
                            [self.merged,self.average_loss,self.average_acc_500,self.average_acc],
                                feed_dict = {
                                        self.is_training : True
                                }                            
                            ) 
                    writer_train.add_summary(s, i)
                    
                    
                    sess.run(
                            [self.train_step]
                            )
                        

                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy:%f'%(self.train_database.epoch,i,self.netLoss,self.netAccuracy))
                    print('[Train] Epoch:%i Iter:%i Loss:%f, Accuracy 500 feat:%f'%(self.train_database.epoch,i,self.netLoss,self.netAccuracy500))


                    
                    if i % 5000 == 0:
                        saver.save(sess, os.path.join(self.save_dir,'%06i.ckpt'%i)) 
                    
                except KeyboardInterrupt:
                    print('Interrupt detected. Ending...')
                    break
                
            saver.save(sess, os.path.join(self.save_dir,'final.ckpt')) 
            print('Model saved')


        
class database_module(object):
    def __init__(
                self,
                image_source_dir,
                database_file,
                batch,
                input_size,
                numclasses,
                shuffle = False
            ):
        
        self.image_source_dir = image_source_dir
        self.database_file = database_file
        self.batch = batch
        self.input_size = input_size
        self.numclasses = numclasses
        self.shuffle = shuffle

        self.load_data_list()
        
    def load_data_list(self):
        with open(self.database_file,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
            
        self.data_paths = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines]
        self.data_labels2 = [int(x.split(' ')[1]) for x in lines]
        self.data_labels3 = [int(x.split(' ')[2]) for x in lines]
        self.data_labels1 = [int(x.split(' ')[3]) for x in lines]
        self.data_num = len(self.data_paths)
        self.data_idx = np.arange(self.data_num)
        self.cursor = 0
        self.epoch = 0
        self.reset_data_list()
        
    def shuffle_data_list(self):
        np.random.shuffle(self.data_idx)
    
    def reset_data_list(self):

        if self.shuffle:
            print('shuffling')
            print(self.data_idx[0:10])            
            np.random.shuffle(self.data_idx)
            print(self.data_idx[0:10])
        self.cursor = 0
        
    def read_batch(self):
        img = []
        lbl1 = []
        lbl2 = []
        lbl3 = []
        while len(img) < self.batch:
            try:
            
                im = cv2.imread(self.data_paths[self.data_idx[self.cursor]])
                if im is None:
                   im = cv2.cvtColor(np.asarray(Image.open(self.data_paths[self.data_idx[self.cursor]]).convert('RGB')),cv2.COLOR_RGB2BGR)
                im = cv2.resize(im,(self.input_size[0:2]))
                if np.ndim(im) == 2:
                    img.append(cv2.cvtColor(im,cv2.COLOR_GRAY2RGB))
                else:
                    img.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                lbl1.append(self.data_labels1[self.data_idx[self.cursor]])
                lbl2.append(self.data_labels2[self.data_idx[self.cursor]])
                lbl3.append(self.data_labels3[self.data_idx[self.cursor]])
            except:
                pass
            
            self.cursor += 1
            if self.cursor >= self.data_num:
                self.reset_data_list()
                self.epoch += 1
        
        img = np.asarray(img,dtype=np.float32)/255.0
        lbl1 = np.asarray(lbl1,dtype=np.int32)
        lbl2 = np.asarray(lbl2,dtype=np.int32)
        lbl3 = np.asarray(lbl3,dtype=np.int32)
        
        return (img,lbl1,lbl2,lbl3)   
    
            
        
    
    
    
    
    
    
    
    
    
    
    
    
