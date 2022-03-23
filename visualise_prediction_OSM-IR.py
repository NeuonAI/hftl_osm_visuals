# Neuon AI - PlantCLEF 2021

"""
current architecture: inception resnet v2
- change network architecture accordingly
"""

import sys
sys.path.append("PATH_TO_TF_SLIM") # path to /models/research/slim
import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim
import numpy as np
import cv2
from nets.inception_resnet_v2 import inception_resnet_v2
from nets import inception_utils
import os
from six.moves import cPickle
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import glob
from bs4 import BeautifulSoup

# ----- Directories ----- #
photo_dir = "PATH_TO_PlantCLEF2020TrainingData/photo"
hpa_dir = "PATH_TO_PlantCLEF2020TrainingData/herbarium_photo_associations"
herbarium_dir = "PATH_TO_PlantCLEF2020TrainingData/herbarium"


checkpoint_model = "checkpoints/OSM/best.ckpt"
savefig_dir = "PATH_TO_SAVED_FIGURES_DIR"

prediction_pkl = "prediction_pkl/osm_run14_best_prediction_eliminated.pkl"
species_map_csv = "list/species_label_plantclef2021.csv"


list_of_test_samples = ["figures/test_sample.jpg"] # note: insert test sample paths here
        

    
# ----- Species map info ----- #
species_df = pd.read_csv(species_map_csv, sep=',')
species_label = species_df['train label'].to_list()
species_folder = species_df['class id'].to_list()
species_name = species_df['species'].to_list()

# ----- Read pkl file ----- #
with open(prediction_pkl,'rb') as fid:
	pred_dict = cPickle.load(fid)

# ----- Get MRR Score ----- #    
def get_rank(dict_value):
    prob = dict_value['prob']
    label = dict_value['label']
    
    idx = np.argsort(prob)[::-1]
    
    np.argmax(prob) == label
    
    rank_i = np.squeeze(np.where(idx==label)) + 1
    
    return rank_i

ranks = np.asarray([get_rank(value) for key,value in pred_dict.items()])
mrr = np.sum((1/ranks))/len(pred_dict)
print("MRR score:", mrr)

test_paths = []
accuracy_correct = []
accuracy_incorrect = []

# ----- Get prediction results ----- #
for key, value in pred_dict.items():

    label = value['label']    
    prob = value['prob']
        
    if key in list_of_test_samples:
        top1_label = np.argsort(prob)[-1:][::-1]
        top1_label = top1_label[0]
               
        test_paths.append(key)
                
        #   Classify correct / incorrect
        if int(label) == int(top1_label):
            accuracy_correct.append((key,label,prob))
        else:
            accuracy_incorrect.append((key,label,prob))
            

print("Accuracy:", len(accuracy_correct), "/", len(test_paths), "=", len(accuracy_correct) / len(test_paths))


#   Get high confidence samples
accuracy_correct_labels = []
accuracy_correct_prob = []  # top1 probabilities
accuracy_correct_probabilities = []
accuracy_correct_paths = []
for pred in accuracy_correct:
    probabilities = pred[2]
    top1_high_idx = np.squeeze(np.argsort(probabilities)[-1:][::-1])
    prob = probabilities[top1_high_idx]
    accuracy_correct_prob.append(prob)
    accuracy_correct_labels.append(pred[1])
    accuracy_correct_probabilities.append(probabilities)
    accuracy_correct_paths.append(pred[0])
    

sorted_accuracy_correct_idx = np.argsort(accuracy_correct_prob)[::-1] # descending order
sorted_accuracy_correct_prob = [accuracy_correct_prob[x] for x in sorted_accuracy_correct_idx]
sorted_accuracy_correct_probabilities = [accuracy_correct_probabilities[x] for x in sorted_accuracy_correct_idx]
sorted_accuracy_correct_labels = [accuracy_correct_labels[x] for x in sorted_accuracy_correct_idx]
sorted_accuracy_correct_paths = [accuracy_correct_paths[x] for x in sorted_accuracy_correct_idx]

#   Get unique high confidence samples
topN_high_confidence = [] 
for path, label, probabilities in zip(sorted_accuracy_correct_paths,
                                      sorted_accuracy_correct_labels,
                                      sorted_accuracy_correct_probabilities):
    

    topN_high_confidence.append((path,label,probabilities))
        

#   Get low confidence samples
accuracy_incorrect_labels = []
accuracy_incorrect_prob = []  # top1 probabilities
accuracy_incorrect_probabilities = []
accuracy_incorrect_paths = []
for pred in accuracy_incorrect:
    probabilities = pred[2]
    top1_low_idx = np.squeeze(np.argsort(probabilities)[-1:][::-1])
    prob = probabilities[top1_low_idx]
    accuracy_incorrect_prob.append(prob)
    accuracy_incorrect_labels.append(pred[1])
    accuracy_incorrect_probabilities.append(probabilities)
    accuracy_incorrect_paths.append(pred[0])
    

sorted_accuracy_incorrect_idx = np.argsort(accuracy_incorrect_prob) # ascending order
sorted_accuracy_incorrect_prob = [accuracy_incorrect_prob[x] for x in sorted_accuracy_incorrect_idx]
sorted_accuracy_incorrect_probabilities = [accuracy_incorrect_probabilities[x] for x in sorted_accuracy_incorrect_idx]
sorted_accuracy_incorrect_labels = [accuracy_incorrect_labels[x] for x in sorted_accuracy_incorrect_idx]
sorted_accuracy_incorrect_paths = [accuracy_incorrect_paths[x] for x in sorted_accuracy_incorrect_idx]

#   Get unique low confidence samples
topN_low_confidence = [] 
for path, label, probabilities in zip(sorted_accuracy_incorrect_paths,
                                      sorted_accuracy_incorrect_labels,
                                      sorted_accuracy_incorrect_probabilities):
    

    topN_low_confidence.append((path,label,probabilities))
    
     
        

# ----- Network hyperparameters ----- #
global_batch = 6 # global_batch * 5 = actual batch
batch = 60
numclasses = 997
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

def data_in_train():
    return tf.map_fn(fn = train_preproc,elems = ims,dtype=np.float32)      

def data_in_test():
    return tf.map_fn(fn = test_preproc,elems = ims,dtype=np.float32)



data_in = tf.cond(
        is_training,
        true_fn = data_in_train,
        false_fn = data_in_test
        )




def normalise_embedding(sample_embedding):
 
    
    #   normalise the values
    norm_min_whole = sample_embedding.min(keepdims=True)
    norm_max_whole = sample_embedding.max(keepdims=True)
    sample_embedding_normalised = (sample_embedding - norm_min_whole)/(norm_max_whole - norm_min_whole)


    return sample_embedding_normalised
   


def get_highest_activated_map(normalised_last_layer):
    
    total_maps = []
    for n in range(normalised_last_layer.shape[2]):
        feature_emb = normalised_last_layer[:,:,n]
        sum_of_emb = np.sum(feature_emb)
        total_maps.append(sum_of_emb)
    
    #   sort activated maps
    sorted_activated_maps = sorted(((value, index) for index, value in enumerate(total_maps)), reverse=True)
    highest_activated_map = sorted_activated_maps[0][1] 
    
    return highest_activated_map



# ----- Construct network ----- #
with slim.arg_scope(inception_utils.inception_arg_scope()):
    logits,endpoints = inception_resnet_v2(data_in, 
                                num_classes=numclasses,
                                is_training=is_training)


    logits_family = slim.fully_connected(endpoints['PreLogitsFlatten'],151,activation_fn=None,
                                scope='Family')
    logits_genus = slim.fully_connected(endpoints['PreLogitsFlatten'],508,activation_fn=None,
                                scope='Genus')    
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
        
feat_species = tf.nn.softmax(logits)
feat_family = tf.nn.softmax(logits_family)
feat_genus = tf.nn.softmax(logits_genus)

    
variables_to_restore = slim.get_variables_to_restore()
restorer = tf.train.Saver(variables_to_restore)

var_list = tf.train.list_variables(checkpoint_model)


sample_im = ims * 1

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
print(f"[{datetimestr()}] Start process")
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, checkpoint_model)
    

    image_list = []
    image_path_list = []
    test_embedding_list = []
    last_layer_list = []
    
    # note: change to high or low confidence samples 
    topN_confidence_samples = [x[0] for x in topN_high_confidence]
    topN_confidence_labels = [x[1] for x in topN_high_confidence]
    topN_confidence_probs = [x[2] for x in topN_high_confidence]

#    topN_confidence_samples = [x[0] for x in topN_low_confidence]
#    topN_confidence_labels = [x[1] for x in topN_low_confidence]
#    topN_confidence_probs = [x[2] for x in topN_low_confidence]  
    
        
    files = topN_confidence_samples
    
    current_class_files = files
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
        

        field_embedding, last_layer_embedding = sess.run(
                [
                feat_500,
                endpoints['Conv2d_7b_1x1'],  
                ],
                feed_dict = {
                            tf_filepath2:paths,   
                            is_training : False,
                            is_train : False
                        }
                )
      
        

        images = np.reshape(ret,(global_batch,10,299,299,3))
           
        field_emb = np.reshape(field_embedding,(global_batch,10,-1))
        average_corner_crops = np.mean(field_emb,axis=1)
        
        last_layer_emb_shape = last_layer_embedding.shape
        last_layer_emb = np.reshape(last_layer_embedding, (global_batch,10,last_layer_emb_shape[1],last_layer_emb_shape[2],-1))        

        
        
        if n == (iter_run - 1):   
            for i,q in enumerate(images[0:(global_batch-padded)]):
                image_list.append(q)    
            for i,p in enumerate(paths[0:(global_batch-padded)]):
                image_path_list.append(p)                 
            for i,a in enumerate(average_corner_crops[0:(global_batch-padded)]):
                test_embedding_list.append(a.reshape(1,500))  
            for i,b in enumerate(last_layer_emb[0:(global_batch-padded)]):
                last_layer_list.append(b)                 
                c += 1
        else:  
            for i,q in enumerate(images):
                image_list.append(q) 
            for i,p in enumerate(paths):
                image_path_list.append(p)                
            for i,a in enumerate(average_corner_crops):
                test_embedding_list.append(a.reshape(1,500)) 
            for i,b in enumerate(last_layer_emb):              
                last_layer_list.append(b)                 
                c += 1
        


    #   Iterate top N samples
    counter = 0
    for sample, label, prob in zip(topN_confidence_samples,
                                               topN_confidence_labels,
                                               topN_confidence_probs):


        class_id_species = species_name[label]
        fig, axs = plt.subplots(nrows=20, ncols=11,figsize=(17, 34)) 
      
        [axi.set_axis_off() for axi in axs.ravel()]  


        # Class ID
        ori_img = cv2.imread(sample)
        ori_img_resized = cv2.resize(ori_img, (299,299))
        class_folder_main = species_folder[label]
        axw = axs[0, 0]
        axw.text(0, 0.3, "Species: " + class_id_species, verticalalignment='center', fontweight='bold', fontsize=22, transform=axw.transAxes)            
        
        plt.sca(axs[1, 0])
               
        plt.imshow(cv2.cvtColor(ori_img_resized, cv2.COLOR_BGR2RGB))      
       
    
        #   Get corner crops + Draw colour maps
        row_layer = 2                   
        current_image = image_list[counter]
        current_last_layer = last_layer_list[counter]
        current_filepath = image_path_list[counter]
        current_file = current_filepath.split("\\")[-1]
        current_filename = current_file.strip(".jpg")
        normalised_current_last_layer = normalise_embedding(current_last_layer)

        #   Get highest activated centred image
        centre_image = current_image[4]
        normalised_centred_last_layer = normalised_current_last_layer[4]
        centred_highest_activated_map = get_highest_activated_map(normalised_centred_last_layer)

        
        #   Get 10 corner crops
        for j in range(len(current_image)):
            cropped_image = current_image[j] 
            plt.sca(axs[row_layer, j])
            plt.title("Crop: {}".format(j + 1), fontweight='bold', fontsize=14) 
            plt.imshow(cropped_image) 
            
            #   Get highest activated map
            normalised_last_layer = normalised_current_last_layer[j]  
            highest_activated_map = get_highest_activated_map(normalised_last_layer)            
            
            normalised_highest_cropped_last_layer = normalised_last_layer[:,:,highest_activated_map]
            normalised_highest_cropped_last_layer_resized = cv2.resize(normalised_highest_cropped_last_layer, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
            plt.sca(axs[row_layer + 1, j])
            plt.title("Index: {}".format(highest_activated_map), fontweight='bold', fontsize=10)         
            plt.imshow(normalised_highest_cropped_last_layer_resized, cmap='jet') 


        #   Get averaged  
        averaged_corner_crop_image = np.mean(current_image, axis=0)
        plt.sca(axs[row_layer, j + 1])
        plt.title("Averaged", fontweight='bold', fontsize=14) 
        plt.imshow(averaged_corner_crop_image)
        
        #   Get highest activated map            
        normalised_averaged_cropped_last_layer = np.mean(normalised_current_last_layer, axis=0)
        highest_activated_map = get_highest_activated_map(normalised_averaged_cropped_last_layer)            
                
        normalised_highest_averaged_cropped_last_layer = normalised_averaged_cropped_last_layer[:,:,highest_activated_map]                      
        normalised_highest_averaged_cropped_last_layer_resized = cv2.resize(normalised_highest_averaged_cropped_last_layer, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        plt.sca(axs[row_layer + 1, j + 1])
        
       
        plt.title("Index: {}".format(highest_activated_map), fontweight='bold', fontsize=10)       
        plt.imshow(normalised_highest_averaged_cropped_last_layer_resized)  

     
        
        #   Get top 5 prediction
        topR_prediction_samples = []
        topR_prediction_labels = np.argsort(prob)[-5:][::-1]
        topR_prediction_probabilities = [prob[x] for x in topR_prediction_labels]
        
        sorted_prediction_probabilities_idx = np.argsort(prob)[::-1]
        sorted_prediction_probabilities_idx_list = sorted_prediction_probabilities_idx.tolist()
        actual_class_rank = sorted_prediction_probabilities_idx_list.index(label) + 1
        
        for lbl in topR_prediction_labels: 
            
            lbl_folder = species_folder[species_label.index(lbl)]
            lbl_folder = str(lbl_folder)
            
            #   Image dir
            lbl_photo_dir = os.path.join(photo_dir,lbl_folder)
            lbl_hpa_dir = os.path.join(hpa_dir,lbl_folder)
            lbl_herbarium_dir = os.path.join(herbarium_dir,lbl_folder)
            
            random_sample_path = []
            #   Check class id path exists
            if os.path.exists(lbl_photo_dir):
                photo_jpg_samples = (glob.glob(lbl_photo_dir + "\*.jpg"))

                samples = photo_jpg_samples
                for sample in samples:
                    random_sample_path.append(sample)
                    if len(random_sample_path) == 11:
                        break

            if (os.path.exists(lbl_hpa_dir)) and (len(random_sample_path) < 11):
                hpa_jpg_samples = (glob.glob(lbl_hpa_dir + "\*.jpg"))

                samples = []
                for path in hpa_jpg_samples:
                     #   Check if sample is herbarium
                    filepath = path.strip(".jpg")
                    filepath_xml = filepath + ".xml"
                    
                    with open(filepath_xml) as fxml:
                        soup = BeautifulSoup(fxml, 'xml')
                        phototype = soup.find_all('PhotoType')[0].get_text()
                        if phototype == "Herbarium":
                            pass
                        else:
                            samples.append(path)
                                                    
                                   

                for sample in samples:
                    random_sample_path.append(sample)
                    if len(random_sample_path) == 11:
                        break

            if (os.path.exists(lbl_herbarium_dir)) and (len(random_sample_path) < 11):
                h_jpg_samples = (glob.glob(lbl_herbarium_dir + "\*.jpg"))

                samples = h_jpg_samples
                for sample in samples:
                    random_sample_path.append(sample)
                    if len(random_sample_path) == 11:
                        break

            
            topR_prediction_samples.append(random_sample_path)
         
        #   Get top R prediction corner crops and last embs
        topR_image_list = []
        topR_image_path_list = []
        topR_last_layer_list_field = []
        topR_last_layer_list_herbarium = []
        topR_current_class_files = []
        
        #   Get all prediction samples
        for classfiles in topR_prediction_samples:
            for file in classfiles:                    
                topR_current_class_files.append(file)
        
        iter_run = len(topR_current_class_files)//global_batch
        
        print(f"[{datetimestr()}] Files:{len(topR_current_class_files)}")
        
        if len(topR_current_class_files) > (iter_run * global_batch):            
            iter_run += 1
            padded = (iter_run * global_batch) - len(topR_current_class_files) 
            topR_current_class_files = topR_current_class_files + ([topR_current_class_files[0]] * padded)
        else:
            padded = 0
        

        for n in range(iter_run):

            topR_paths = topR_current_class_files[n*global_batch:(n*global_batch)+global_batch] 
            topR_ret = sess.run(sample_im,feed_dict = {
                    tf_filepath2:topR_paths})
            
            #   Field stream
            topR_last_layer_embedding_field = sess.run(
                        endpoints['Conv2d_7b_1x1'],  
                        feed_dict = {
                                        tf_filepath2:topR_paths,
                                        is_training : False,
                                        is_train : False
                                }
                    )
            
            #   Herbarium stream
            topR_last_layer_embedding_herbarium = sess.run(
                        endpoints['Conv2d_7b_1x1'], 
                        feed_dict = {
                                        tf_filepath2:topR_paths,
                                        is_training : False,
                                        is_train : False
                                }
                    )            
            
            topR_last_layer_embedding_shape_field = topR_last_layer_embedding_field.shape
            topR_last_layer_emb_field = np.reshape(topR_last_layer_embedding_field, (global_batch,10,topR_last_layer_embedding_shape_field[1],topR_last_layer_embedding_shape_field[2],-1))        

            topR_last_layer_embedding_shape_herbarium = topR_last_layer_embedding_herbarium.shape
            topR_last_layer_emb_herbarium = np.reshape(topR_last_layer_embedding_herbarium, (global_batch,10,topR_last_layer_embedding_shape_herbarium[1],topR_last_layer_embedding_shape_herbarium[2],-1))             
            
            topR_images = np.reshape(topR_ret,(global_batch,10,299,299,3))
            
            
            if n == (iter_run - 1):   
                for i,q in enumerate(topR_images[0:(global_batch-padded)]):
                    topR_image_list.append(q)    
                for i,p in enumerate(topR_paths[0:(global_batch-padded)]):
                    topR_image_path_list.append(p)                 
                for i,b in enumerate(topR_last_layer_emb_field[0:(global_batch-padded)]):
                    topR_last_layer_list_field.append(b) 
                for i,b in enumerate(topR_last_layer_emb_herbarium[0:(global_batch-padded)]):
                    topR_last_layer_list_herbarium.append(b)                     

            else:  
                for i,q in enumerate(topR_images):
                    topR_image_list.append(q) 
                for i,p in enumerate(topR_paths):
                    topR_image_path_list.append(p)                
                for i,b in enumerate(topR_last_layer_emb_field):              
                    topR_last_layer_list_field.append(b) 
                for i,b in enumerate(topR_last_layer_emb_herbarium):              
                    topR_last_layer_list_herbarium.append(b)                      

        
        row_layer += 2
        predR_counter = 0
        predid_end = 0
        #   Get top R corner crops + Draw colour maps
        for predid, predid_samples, predid_prob in zip(topR_prediction_labels, topR_prediction_samples,topR_prediction_probabilities):
            predid_len = len(predid_samples)
            predid_end += predid_len
            predid_start = predid_end - predid_len
            
            column = 0 
            row_layer += 1
            predid_species = species_name[predid]
            
            
            for im, im_path, im_last_layer_field, im_last_layer_herbarium in zip(topR_image_list[predid_start:predid_end],
                                                                                 topR_image_path_list[predid_start:predid_end],
                                                                                 topR_last_layer_list_field[predid_start:predid_end],
                                                                                 topR_last_layer_list_herbarium[predid_start:predid_end]):
                
                #   Check if sample is herbarium
                filepath = im_path.strip(".jpg")
                filepath_xml = filepath + ".xml"
                
                with open(filepath_xml) as fxml:

                    soup = BeautifulSoup(fxml, 'xml')
                    phototype = soup.find_all('PhotoType')[0].get_text()
                    if phototype == "Herbarium":
                        im_last_layer = im_last_layer_herbarium
                        layer_title = "Index: {}"
                        top_title = "Index: {}"
                    else:
                        im_last_layer = im_last_layer_field
                        layer_title = "Index: {}"
                        top_title = "Index: {}"
                
                normalised_im_last_layer = normalise_embedding(im_last_layer)
     
                pred_class_id = topR_prediction_labels[predR_counter]
                pred_class_name = species_folder[pred_class_id]
                
                #   Check class id in predictions
                if int(class_folder_main) == int(species_folder[pred_class_id]):
                    title_color = "green"
                else:
                    title_color = "black"
                    
                if column == 0:
                    axw = axs[row_layer - 1, 0]
                    axw.text(0.1, 0, "Top-{} prediction: {}; {}".format(predR_counter+1,predid_species, round(predid_prob,4)), horizontalalignment='left', fontweight='bold', fontsize=18, transform=axw.transAxes, color=title_color)     

                
                
                #   Get 10 samples (centre)
                current_image = im[4]
                normalised_im_current_last_layer = normalised_im_last_layer[4]
                plt.sca(axs[row_layer, column])
                plt.imshow(current_image)             
        
                #   Get highest activated map 
                highest_activated_map = get_highest_activated_map(normalised_im_current_last_layer)            
                
                normalised_highest_im_last_layer = normalised_im_current_last_layer[:,:,highest_activated_map]
                normalised_highest_im_last_layer_resized = cv2.resize(normalised_highest_im_last_layer, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
                plt.sca(axs[row_layer + 1, column])                   
                plt.title(top_title.format(highest_activated_map), fontweight='bold', fontsize=10)            
                plt.imshow(normalised_highest_im_last_layer_resized, cmap='jet') 
                
                
                column += 1 
            row_layer += 2
            
                
            predR_counter += 1
            
        counter += 1   

        axw = axs[19, 5]
        axw.text(0.5, 0.5, "(B)", horizontalalignment='center', fontweight='bold', fontsize=20, transform=axw.transAxes) 

        suptitle = 'OSM Inception-ResNet-v2\nConv2d_7b_1x1 layer\n Actual class rank: ' + str(actual_class_rank) 
        plt.sca(axs[0, 10])
        plt.title(suptitle, fontweight='bold', fontsize=22, loc='right')
    
     
        plt.tight_layout()    
        plt.savefig(os.path.join(savefig_dir, "OSM_IR_prediction_" + str(counter) + "_" + str(class_folder_main)) + "_" + current_filename + ".jpg", bbox_inches='tight')
        plt.show()
        
    
    print(f"[{datetimestr()}] Counter:{c}")        
        

 



        

       