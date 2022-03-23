# Neuon AI - PlantCLEF 2021

"""
Network module for HFTL
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
from nets.inception_v4 import inception_v4
from nets import inception_utils
from PIL import Image
import cv2
import random

class network_module(object):
    def __init__(self,
                 batch,
                 iterbatch,
                 numclasses1,
                 numclasses2,
                 image_dir_parent_train,
                 image_dir_parent_test,
                 train_file1,
                 train_file2,
                 test_file1,
                 test_file2,
                 input_size,
                 checkpoint_model1,
                 checkpoint_model2,
                 learning_rate,
                 save_dir,
                 max_iter,
                 val_freq,
                 val_iter):
        
        self.batch = batch
        self.iterbatch = iterbatch
        self.image_dir_parent_train = image_dir_parent_train
        self.image_dir_parent_test = image_dir_parent_test
        self.train_file1 = train_file1
        self.train_file2 = train_file2
        self.test_file1 = test_file1
        self.test_file2 = test_file2
        self.input_size = input_size
        self.numclasses1 = numclasses1
        self.numclasses2 = numclasses2
        self.checkpoint_model1 = checkpoint_model1
        self.checkpoint_model2 = checkpoint_model2
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.max_iter = max_iter
        self.val_freq = val_freq
        self.val_iter = val_iter
        

                
        # ----- Database module ----- #
        self.train_database = database_module(
                image_source_dir = self.image_dir_parent_train,
                database_file1 = self.train_file1,
                database_file2 = self.train_file2,
                batch = self.batch,
                input_size = self.input_size,
                numclasses1 = self.numclasses1,
                numclasses2 = self.numclasses2,
                shuffle = True)

        self.test_database = database_module(
                image_source_dir = self.image_dir_parent_test,
                database_file1 = self.test_file1,
                database_file2 = self.test_file2,
                batch = self.batch,
                input_size = self.input_size,
                numclasses1 = self.numclasses1,
                numclasses2 = self.numclasses2,
                shuffle = True)
        

           
        
        # ----- Tensors ------ #
        x1 = tf.placeholder(tf.float32,(None,) + self.input_size)
        x2 = tf.placeholder(tf.float32,(None,) + self.input_size)
        herbarium_embs = tf.placeholder(tf.float32)
        field_embs = tf.placeholder(tf.float32)
        feat_concat = tf.placeholder(tf.float32, shape=[None, 500])
        lbl_concat = tf.placeholder(tf.float32)
        y1 = tf.placeholder(tf.int32, (None,))
        y2 = tf.placeholder(tf.int32, (None,))
        self.is_training = tf.placeholder(tf.bool)
        is_train = tf.placeholder(tf.bool, name="is_training")
               
        
        
     
        # ----- Image pre-processing methods ----- #      
        train_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=True)
        
        test_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=False) 
        
        def data_in_train1():
            return tf.map_fn(fn = train_preproc,elems = x1,dtype=np.float32)
        
        def data_in_test1():
            return tf.map_fn(fn = test_preproc,elems = x1,dtype=np.float32)
        
        def data_in_train2():
            return tf.map_fn(fn = train_preproc,elems = x2,dtype=np.float32)
        
        def data_in_test2():
            return tf.map_fn(fn = test_preproc,elems = x2,dtype=np.float32)
        
        data_in1 = tf.cond(
                self.is_training,
                true_fn = data_in_train1,
                false_fn = data_in_test1
                )
        
        data_in2 = tf.cond(
                self.is_training,
                true_fn = data_in_train2,
                false_fn = data_in_test2
                )
        
        
        
        # ----- Network 1 construction ----- #
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits,endpoints = inception_v4(data_in1,
                                            num_classes=self.numclasses1,
                                            is_training=self.is_training,
                                            scope='herbarium'
                                            )
            

            herbarium_embs = endpoints['PreLogitsFlatten']

            

            herbarium_bn = tf.layers.batch_normalization(herbarium_embs, training=is_train)

            herbarium_feat = tf.contrib.layers.fully_connected(
                            inputs=herbarium_bn,
                            num_outputs=500,
                            activation_fn=None,
                            normalizer_fn=None,
                            trainable=True,
                            scope='herbarium'                            
                    )
            
            herbarium_feat = tf.math.l2_normalize(
                                                herbarium_feat,
                                                axis=1      
                                            )

        
        
        
        # ----- Network 2 construction ----- #        
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits2,endpoints2 = inception_v4(data_in2,
                                            num_classes=self.numclasses2,
                                            is_training=self.is_training,
                                            scope='field'
                                            )
            

            field_embs = endpoints2['PreLogitsFlatten']      

            
            field_bn = tf.layers.batch_normalization(field_embs, training=is_train)

            field_feat = tf.contrib.layers.fully_connected(
                            inputs=field_bn,
                            num_outputs=500,
                            activation_fn=None,
                            normalizer_fn=None,
                            trainable=True,
                            scope='field'                            
                    )            
            
            field_feat = tf.math.l2_normalize(
                                    field_feat,
                                    axis=1      
                                )
           
        
        
        feat_concat = tf.concat([herbarium_feat, field_feat], 0)
        lbl_concat = tf.concat([y1, y2], 0)


        # ----- Get all variables ----- #
        self.variables_to_restore = tf.trainable_variables()
        
        self.variables_bn = [k for k in self.variables_to_restore if k.name.startswith('batch_normalization')]
        self.variables_herbarium = [k for k in self.variables_to_restore if k.name.startswith('herbarium')]
        self.variables_field = [k for k in self.variables_to_restore if k.name.startswith('field')]



        # ----- New variable list ----- #        
        self.var_list_front = self.variables_herbarium[0:-10] + self.variables_field[0:-10]        
        self.var_list_end = self.variables_herbarium[-10:] + self.variables_field[-10:] + self.variables_bn
        self.var_list_train = self.var_list_front + self.var_list_end

        
        
        
        # ----- Network losses ----- #
        with tf.name_scope("loss_calculation"): 
            with tf.name_scope("triplets_loss"):
                self.triplets_loss = tf.reduce_mean(
                        tf.contrib.losses.metric_learning.triplet_semihard_loss(
                                labels=lbl_concat, embeddings=feat_concat, margin=1.0))

            with tf.name_scope("L2_reg_loss"):
#                self.regularization_loss = 0.00004 * tf.add_n(tf.losses.get_regularization_losses(scope='InceptionV4'))
                self.regularization_loss = tf.add_n([ tf.nn.l2_loss(v) for v in self.var_list_train]) * 0.00004 
                
            with tf.name_scope("total_loss"):
                self.totalloss = self.triplets_loss + self.regularization_loss
                
                        
            
        # ----- Create update operation ----- #
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)    
        self.vars_ckpt = slim.get_variables_to_restore()
    
        vars_ckpt_herbarium = [k for k in self.vars_ckpt if k.name.startswith('herbarium')]     
        vars_ckpt_field = [k for k in self.vars_ckpt if k.name.startswith('field')]

        
        # ----- Restore model 1 ----- #
        restore_fn1 = slim.assign_from_checkpoint_fn(
            self.checkpoint_model1, vars_ckpt_herbarium[:-2])       

        
        # ----- Restore model 2 ----- #
        restore_fn2 = slim.assign_from_checkpoint_fn(
            self.checkpoint_model2, vars_ckpt_field[:-2]) 
        

       
        # ----- Training scope ----- #       
        with tf.name_scope("train"):
            loss_accumulator = tf.Variable(0.0, trainable=False)
            
            self.collect_loss = loss_accumulator.assign_add(self.totalloss)
                        
            self.average_loss = tf.cond(self.is_training,
                                        lambda: loss_accumulator / self.iterbatch,
                                        lambda: loss_accumulator / self.val_iter)
            
            self.zero_op_loss = tf.assign(loss_accumulator,0.0)


            
            # ----- Separate vars ----- #
            self.accum_train_front = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_front] 
            self.accum_train_end = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_end]                                               
        
            self.zero_ops_train_front = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_front]
            self.zero_ops_train_end = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_end]
            
            # ----- Set up optimizer / Compute gradients ----- #
            with tf.control_dependencies(self.update_ops):

                optimizer = tf.train.AdamOptimizer(self.learning_rate * 0.1)
                
                # create another optimizer
                optimizer_end_layers = tf.train.AdamOptimizer(self.learning_rate)
                
                
                # compute gradient with an other list of var_list
                gradient1 = optimizer.compute_gradients(self.totalloss,self.var_list_front)
                gradient2 = optimizer_end_layers.compute_gradients(self.totalloss,self.var_list_end)
              


                gradient_only_front = [gc[0] for gc in gradient1]
                gradient_only_front,_ = tf.clip_by_global_norm(gradient_only_front,1.25)
                
                gradient_only_back = [gc[0] for gc in gradient2]
                gradient_only_back,_ = tf.clip_by_global_norm(gradient_only_back,1.25)
                
               
                self.accum_train_ops_front = [self.accum_train_front[i].assign_add(gc) for i,gc in enumerate(gradient_only_front)]
            
                self.accum_train_ops_end = [self.accum_train_end[i].assign_add(gc) for i,gc in enumerate(gradient_only_back)]



            # ----- Apply gradients ----- #
            self.train_step_front = optimizer.apply_gradients(
                    [(self.accum_train_front[i], gc[1]) for i, gc in enumerate(gradient1)])
      
            self.train_step_end = optimizer_end_layers.apply_gradients(
                    [(self.accum_train_end[i], gc[1]) for i, gc in enumerate(gradient2)])
            
            

        # ----- Global variables ----- #
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]             
        var_list += bn_moving_vars
        
        
        
        # ----- Create saver ----- #
        saver = tf.train.Saver(var_list=var_list, max_to_keep=0)

        tf.summary.scalar('loss',self.average_loss) 
        self.merged = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'loss')])
        
        
        # ----- Tensorboard writer--- #
        tensorboar_dir = 'tensorboard'

        writer_train = tf.summary.FileWriter(tensorboar_dir+'/train')
        writer_test = tf.summary.FileWriter(tensorboar_dir+'/test')

       

        # ----- Create session 1 ----- #
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            writer_train.add_graph(sess.graph)
            writer_test.add_graph(sess.graph)
            
            restore_fn1(sess)
            restore_fn2(sess)

            
            for i in range(self.max_iter+1):
                try:
                    sess.run(self.zero_ops_train_front)
                    sess.run(self.zero_ops_train_end)
                    sess.run([self.zero_op_loss])                    
                    
                    
                    
                    # ----- Validation ----- #
                    if i % self.val_freq == 0:                        
                        print('Start:%f'%sess.run(loss_accumulator))
                        for j in range(self.val_iter):
                            img1,img2,lbl1, lbl2 = self.test_database.read_batch()
                            sess.run(
                                        self.collect_loss,
                                        feed_dict = {x1 : img1,
                                                     x2 : img2,
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     self.is_training : False,
                                                     is_train : False
                                        }                                  
                                    )
                            print('[%i]:%f'%(j,sess.run(loss_accumulator)))
                        print('End:%f'%sess.run(loss_accumulator))  
                        s,self.netLoss = sess.run(                        
                                [self.merged,self.average_loss],
                                    feed_dict = {
                                            self.is_training : False
                                    }                            
                                ) 
                        
                        writer_test.add_summary(s, i)
                        print('[Valid] Epoch:%i Iter:%i Loss:%f'%(self.test_database.epoch,i,self.netLoss))

                        sess.run([self.zero_op_loss])
                        


                    # ----- Train ----- #
                    for j in range(self.iterbatch):
                        img1,img2,lbl1,lbl2 = self.train_database.read_batch()
    
                        sess.run(
                                    [self.collect_loss,self.accum_train_ops_front,self.accum_train_ops_end],
                                    feed_dict = {x1 : img1, 
                                                 x2 : img2,
                                                 y1 : lbl1,
                                                 y2 : lbl2,
                                                 self.is_training : True,
                                                 is_train : True
                                    }                                
                                )
                        
                    s,self.netLoss = sess.run(
                            [self.merged,self.average_loss],
                                feed_dict = {
                                        self.is_training : True
                                }                            
                            ) 
                    writer_train.add_summary(s, i)
                    
                    sess.run([self.train_step_front])
                    sess.run([self.train_step_end])
                        
                    print('[Train] Epoch:%i Iter:%i Loss:%f'%(self.train_database.epoch,i,self.netLoss))

                    
                    if i % 5000 == 0:
                        saver.save(sess, os.path.join(self.save_dir,'%06i.ckpt'%i)) 
                    
                except KeyboardInterrupt:
                    print('Interrupt detected. Ending...')
                    break
                
            # ----- Save model --- #
            saver.save(sess, os.path.join(self.save_dir,'final.ckpt')) 
            print('Model saved')




           
class database_module(object):
    def __init__(
                self,
                image_source_dir,
                database_file1,
                database_file2,
                batch,
                input_size,
                numclasses1,
                numclasses2,
                shuffle = False
            ):
        
        print("Initialising...")
        self.image_source_dir = image_source_dir
        self.database_file1 = database_file1
        self.database_file2 = database_file2
        self.batch = batch
        self.input_size = input_size
        self.numclasses1 = numclasses1
        self.numclasses2 = numclasses2
        self.shuffle = shuffle

        self.load_data_list()
        
    def load_data_list(self):        
        self.database1_dict = {}
        self.database2_dict = {}
        
        # ----- Dataset 1 ----- #
        with open(self.database_file1,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
            
        self.data_paths1 = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines]
        self.data_labels1 = [int(x.split(' ')[3]) for x in lines]

        for key, value in zip(self.data_labels1, self.data_paths1):
            if key not in self.database1_dict:
                self.database1_dict[key] = [] 

            self.database1_dict[key].append(value)
                           
        
        # ----- Dataset 2 ----- #
        with open(self.database_file2,'r') as fid2:
            lines2 = [x.strip() for x in fid2.readlines()]
            
        self.data_paths2 = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines2]
        self.data_labels2 = [int(x.split(' ')[3]) for x in lines2]

        for key, value in zip(self.data_labels2, self.data_paths2):
            if key not in self.database2_dict:
                self.database2_dict[key] = [] 

            self.database2_dict[key].append(value)
        
        
        self.data_num1 = len(self.data_paths1)
        self.data_num2 = len(self.data_paths2)
        self.database1_dict_copy = self.database1_dict
        self.database2_dict_copy = self.database2_dict        

        self.unique_labels = list(set(self.data_labels1).intersection(self.data_labels2))


        self.epoch = 0
        self.reset_data_list()     
        
    
    def reset_data_list(self):
        
        self.database1_dict = {}
        self.database2_dict = {}
        
        # ----- Dataset 1 ----- #
        with open(self.database_file1,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
            
        self.data_paths1 = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines]
        self.data_labels1 = [int(x.split(' ')[3]) for x in lines]

        for key, value in zip(self.data_labels1, self.data_paths1):
            if key not in self.database1_dict:
                self.database1_dict[key] = [] 

            self.database1_dict[key].append(value)
                
            
        
        # ----- Dataset 2 ----- #
        with open(self.database_file2,'r') as fid2:
            lines2 = [x.strip() for x in fid2.readlines()]
            
        self.data_paths2 = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines2]
        self.data_labels2 = [int(x.split(' ')[3]) for x in lines2]

        for key, value in zip(self.data_labels2, self.data_paths2):
            if key not in self.database2_dict:
                self.database2_dict[key] = [] 

            self.database2_dict[key].append(value)
  
        
        self.database1_dict_copy = self.database1_dict
        self.database2_dict_copy = self.database2_dict
        self.unique_labels = list(set(self.data_labels1).intersection(self.data_labels2))


        
    def read_img(self, fp):
        try:
            im = cv2.imread(fp)
        
            if im is None:
               im = cv2.cvtColor(np.asarray(Image.open(fp).convert('RGB')),cv2.COLOR_RGB2BGR)
            im = cv2.resize(im,(self.input_size[0:2]))
        
            if np.ndim(im) == 2:
                im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
                
            else:
                im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            
        except:
            pass


        return im


    def get_random_class(self):
        random_class = random.choices(self.unique_labels, k=1)        
        return random_class    
    
    def get_paths(self, class_i):
        try:
            # Anchor positive
            rand_anchor1 = random.choice(self.database1_dict_copy.get(class_i))
        except:
            rand_anchor1 = None

        try:
            # Anchor negative
            rand_anchor2 = random.choice(self.database2_dict_copy.get(class_i))
        except:
            rand_anchor2 = None

        return rand_anchor1, rand_anchor2
            
    def get_random_anchors(self, class_i, rand_anchor1, rand_anchor2):        
        # Random anchor 1 (positive)
        im = self.read_img(rand_anchor1)
        if len(self.total_filepaths) < self.batch:
            self.img1.append(im)
            self.lbl1.append(class_i)
            self.total_filepaths.append(rand_anchor1)
            
            # Remove anchor from dictionary 1
            for key, value in self.database1_dict_copy.items():
                if key == class_i:
                    if rand_anchor1 in value:
                        value.remove(rand_anchor1)
                        
        # Random anchor 2 (negative)               
        im2 = self.read_img(rand_anchor2)
        if len(self.total_filepaths) < self.batch:
            self.img2.append(im2)
            self.lbl2.append(class_i)
            self.total_filepaths.append(rand_anchor2) 
            
            # Remove anchor from dictionary 2
            for key, value in self.database2_dict_copy.items():
                if key == class_i:
                    if rand_anchor2 in value:
                        value.remove(rand_anchor2)
    

        
        
    def read_batch(self):        
        self.total_filepaths = []
        self.img1 = []
        self.img2 = []
        self.lbl1 = [] 
        self.lbl2 = []

        current_class_labels = []


        while len(self.total_filepaths) < self.batch:
            
            try:
                # Select random class
                class_i = self.get_random_class()[0]

            except:
                # Insufficient data
                print("Resetting data list")
                self.reset_data_list()
                self.epoch += 1
                continue
                

            if class_i not in current_class_labels:
                current_class_labels.append(class_i)

                
            # Check class has 4 samples
            class_i_count1 = self.lbl1.count(class_i)
            class_i_count2 = self.lbl2.count(class_i)
            class_i_count = class_i_count1 + class_i_count2
            if class_i_count >= 4:
                current_class_labels.remove(class_i)
                continue

            

            # Iterate over current labels
            for class_i in current_class_labels:
                # Get anchor paths
                rand_anchor1, rand_anchor2 = self.get_paths(class_i)
                
                if rand_anchor1 is not None and rand_anchor2 is not None:

                    # Get anchor positive and negative
                    self.get_random_anchors(class_i, rand_anchor1, rand_anchor2)
                    
                if rand_anchor1 is None or rand_anchor2 is None:

                    # Remove class from unique list
                    self.unique_labels.remove(class_i)
                    current_class_labels.remove(class_i)
                    continue
                    

                    
                # Check class has 4 samples
                class_i_count1 = self.lbl1.count(class_i)
                class_i_count2 = self.lbl2.count(class_i)
                class_i_count = class_i_count1 + class_i_count2
                if class_i_count >= 4:                    
                    current_class_labels.remove(class_i)

                        
                        
            if len(self.unique_labels) < 4:
                # Insufficient data
                print("Resetting data list")
                self.reset_data_list()
                self.epoch += 1
                continue




        self.img1 = np.asarray(self.img1,dtype=np.float32)/255.0
        self.img2 = np.asarray(self.img2,dtype=np.float32)/255.0
                
            
        return (self.img1, self.img2, self.lbl1, self.lbl2)
    
     

    
    
    
    
    
    
    
    
    
