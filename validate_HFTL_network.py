# Neuon AI - PlantCLEF 2021

"""
current architecture: inception v4
- change network architecture accordingly
"""

import sys
sys.path.append("PATH_TO_SLIM") # path to /models/research/slim
import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim
import numpy as np
from nets.inception_v4 import inception_v4
from nets import inception_utils
import os
from six.moves import cPickle
from sklearn.metrics.pairwise import cosine_similarity
import datetime


# ----- Directories ----- #
image_dir_parent_train = "PATH_TO_PlantCLEF2021TrainingData"
image_dir_parent_test = "PATH_TO_PlantCLEF2021TrainingData"

test_field_file = "list/HFTL/clef2020_known_classes_field_test.txt" # Test Set 1 (with field training data)
#test_field_file = "list/missing_class_sample.txt" # Test Set 2 (without field training data)

checkpoint_model = "PATH_TO_TRAINED_MODEL" # .ckpt

herbarium_dictionary_file = "PATH_TO_SAVED_HERBARIUM_DICTIONARY_FILE" # .pkl

prediction_file = "PATH_TO_SAVE_PREDICTION_PKL_FILE" # .pkl



field_dict = {}
# ----- Load field ----- #
with open(test_field_file,'r') as fid2:
    f_lines = [x.strip() for x in fid2.readlines()]
    
field_paths = [os.path.join(image_dir_parent_test,
                                x.split(' ')[0]) for x in f_lines]
field_labels = [int(x.split(' ')[3]) for x in f_lines]

for key, value in zip(field_labels, field_paths):
    if key not in field_dict:
        field_dict[key] = [] 
    
#    value = value.replace("/", "\\")
    field_dict[key].append(value)

 
    
# ----- Read dictionary pkl file ----- #
with open(herbarium_dictionary_file,'rb') as fid1:
	herbarium_dictionary = cPickle.load(fid1)
    

# ----- Network hyperparameters ----- #
global_batch = 6 # global_batch * 5 = actual batch
batch = 60
numclasses1 = 997
numclasses2 = 10000
input_size = (299,299,3)
img_height, img_width = 299, 299

# ----- Initiate tensors ----- #
x1 = tf.placeholder(tf.float32,(batch,) + input_size)
x2 = tf.placeholder(tf.float32,(batch,) + input_size)
y1 = tf.placeholder(tf.int32,(batch,))
y2 = tf.placeholder(tf.int32,(batch,))
is_training = tf.placeholder(tf.bool)
is_train = tf.placeholder(tf.bool, name="is_training")

tf_filepath2 =  tf.placeholder(tf.string,shape=(global_batch,))

def datetimestr():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

def read_images(p):
    im = tf.io.read_file(p)    
    im =  tf.cast(tf.image.resize_images(tf.image.decode_png(
        im, channels=3, dtype=tf.uint8),(299,299)),tf.float32)
    
    im1 = im[0:260,0:260,:]
    im2 = im[0:260,-260:,:]
    im3 = im[-260:,0:260,:]
    im4 = im[-260:,-260:,:]
    im5 = im[19:279,19:279,:]
    
    im1 =  tf.cast(tf.image.resize_images(im1,(299,299)),tf.float32)
    im2 =  tf.cast(tf.image.resize_images(im2,(299,299)),tf.float32)
    im3 =  tf.cast(tf.image.resize_images(im3,(299,299)),tf.float32)
    im4 =  tf.cast(tf.image.resize_images(im4,(299,299)),tf.float32)
    im5 =  tf.cast(tf.image.resize_images(im5,(299,299)),tf.float32)
    
    im6 = tf.image.flip_left_right(im1)
    im7 = tf.image.flip_left_right(im2)
    im8 = tf.image.flip_left_right(im3)
    im9 = tf.image.flip_left_right(im4)
    im10 = tf.image.flip_left_right(im5)
    
    return tf.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9,im10])

ims = tf.map_fn(fn=read_images,elems=tf_filepath2,dtype=np.float32)
ims = tf.reshape(ims,(batch,)+input_size)/255.0


# ----- Image preprocessing methods ----- #
train_preproc = lambda xi: inception_preprocessing.preprocess_image(
        xi,input_size[0],input_size[1],is_training=True)

test_preproc = lambda xi: inception_preprocessing.preprocess_image(
        xi,input_size[0],input_size[1],is_training=False)  

def data_in_train1():
    return tf.map_fn(fn = train_preproc,elems = ims,dtype=np.float32)      

def data_in_test1():
    return tf.map_fn(fn = test_preproc,elems = ims,dtype=np.float32)

def data_in_train2():
    return tf.map_fn(fn = train_preproc,elems = ims,dtype=np.float32)      

def data_in_test2():
    return tf.map_fn(fn = test_preproc,elems = ims,dtype=np.float32)

data_in1 = tf.cond(
        is_training,
        true_fn = data_in_train1,
        false_fn = data_in_test1
        )

data_in2 = tf.cond(
        is_training,
        true_fn = data_in_train2,
        false_fn = data_in_test2
        )



def match_herbarium_dictionary(test_embedding_list, herbarium_emb_list):
    similarity = cosine_similarity(test_embedding_list, herbarium_emb_list)
        
    k_distribution = []
    # 1 - Cosine
    print("Get probability distribution")
    for sim in similarity:
        new_distribution = []
        for d in sim:
            new_similarity = 1 - d
            new_distribution.append(new_similarity)
        k_distribution.append(new_distribution)
        
    k_distribution = np.array(k_distribution)
        
              
    softmax_list = []
    # Inverse weighting
    for d in k_distribution:
        inverse_weighting = (1/np.power(d,5))/np.sum(1/np.power(d,5))
        softmax_list.append(inverse_weighting)
    
    softmax_list = np.array(softmax_list)    
    
    return softmax_list



# ----- Construct network 1 ----- #
with slim.arg_scope(inception_utils.inception_arg_scope()):
    logits,endpoints = inception_v4(data_in1,
                                num_classes=numclasses1,
                                is_training=is_training,
                                scope='herbarium')

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

# ----- Construct network 2 ----- #     
with slim.arg_scope(inception_utils.inception_arg_scope()):
    logits2,endpoints2 = inception_v4(data_in2,
                                num_classes=numclasses2,
                                is_training=is_training,
                                scope='field')
                            
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

    
variables_to_restore = slim.get_variables_to_restore()
restorer = tf.train.Saver(variables_to_restore)

sample_im = ims * 1

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
print(f"[{datetimestr()}] Start process")
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, checkpoint_model)
    
    counter = 0
    test_embedding_list = []
    prediction_dictionary = {}

    ground_truth_list = []
    filepath_list = []
    
    # ------ Get herbarium dictionary ----- #
    herbarium_emb_list = []
    print(f"[{datetimestr()}] Get herbarium dictionary")
    for herbarium_class, herbarium_emb in herbarium_dictionary.items():
        herbarium_emb_list.append(np.squeeze(herbarium_emb))
    
    herbarium_emb_list = np.array(herbarium_emb_list)
    
    # ----- Iterate each class ----- #
    for key in field_dict.keys():
        print(f"[{datetimestr()}] Key {key}")
        
        current_class_files = field_dict[key]
        iter_run = len(current_class_files)//global_batch
        
        print(f"[{datetimestr()}] Files:{len(current_class_files)}")

        if len(current_class_files) > (iter_run * global_batch):            
            iter_run += 1
            padded = (iter_run * global_batch) - len(current_class_files) 
            current_class_files = current_class_files + ([current_class_files[0]] * padded)
        else:
            padded = 0
        
        c = 0
        for n in range(iter_run):

            paths = current_class_files[n*global_batch:(n*global_batch)+global_batch]            

            ret = sess.run(sample_im,feed_dict = {
                                    tf_filepath2:paths})
            
            sample_embedding = sess.run(
                        field_feat, 
                        feed_dict = {
                                    tf_filepath2:paths,   
                                    is_training : False,
                                    is_train : False
                                }
                    )

            sample_embedding = np.reshape(sample_embedding,(global_batch,10,-1))
            average_corner_crops = np.mean(sample_embedding,axis=1)
            if n == (iter_run - 1):                
                for i,a in enumerate(average_corner_crops[0:(global_batch-padded)]):
                    test_embedding_list.append(a.reshape(1,500)) 
                    ground_truth_list.append(key)
                    filepath_list.append(paths[i])
                    c += 1
            else:            
                for i,a in enumerate(average_corner_crops):
                    test_embedding_list.append(a.reshape(1,500))  
                    ground_truth_list.append(key)
                    filepath_list.append(paths[i])
                    c += 1


                    
        print(f"[{datetimestr()}] Counter:{c}")

    
    
    len_test_embedding_list = len(test_embedding_list)     
    test_embedding_all_crop_list = np.asarray(test_embedding_list)
    test_embedding_all_crop_list = np.reshape(test_embedding_all_crop_list, (len_test_embedding_list,500))
        
    # ----- Iterate sample over herbarium mean class ----- #
    print(f"[{datetimestr()}] Comparing sample embedding with herbarium distance...")
    softmax_all_crop_list = match_herbarium_dictionary(test_embedding_all_crop_list, herbarium_emb_list)
        

    # ----- Get all crops results ----- #
    print(f"[{datetimestr()}] Get top N predictions all crops...")
    for prediction, key, fp in zip(softmax_all_crop_list, ground_truth_list, filepath_list):
        if fp not in prediction_dictionary:
            prediction_dictionary[fp] = {'prob' : [], 'label' : []}
        prediction_dictionary[fp]['prob'] = prediction
        prediction_dictionary[fp]['label'] = key
              
        
 
    # ----- Save prediction file ----- #
    with open(prediction_file,'wb') as fid:
        cPickle.dump(prediction_dictionary,fid,protocol=cPickle.HIGHEST_PROTOCOL)
        print(f"[{datetimestr()}] Pkl 1 file created")


        

       