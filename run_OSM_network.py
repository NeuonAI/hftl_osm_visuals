# Neuon AI - PlantCLEF 2021

from network_module_OSM import network_module as net_module

image_dir_parent_train = "PATH_TO_PlantCLEF2020TrainingData"
image_dir_parent_test = "PATH_TO_PlantCLEF2020TrainingData"

train_file = "list/OSM/clef2021_mixed_class_train.txt"
test_file = "list/OSM/clef2021_mixed_class_test.txt"

checkpoint_model = "PATH_TO_PRETRAINED_SLIM_MODEL_IMAGENET" # inception_v4_2016_09_09/inception_v4.ckpt
checkpoint_save_dir = "PATH_TO_CHECKPOINT_SAVE_DIR"
tensorboard_save_dir = "PATH_TO_TENSORBOARD_SAVE_DIR"

batch = 64
input_size = (299,299,3)
numclasses = 997
learning_rate = 0.0001
iterbatch = 4
max_iter = 5000000
val_freq = 200
val_iter = 100



network = net_module(
        batch = batch,
        iterbatch = iterbatch,
        numclasses = numclasses,
        input_size = input_size,
        image_dir_parent_train = image_dir_parent_train,
        image_dir_parent_test = image_dir_parent_test,
        train_file = train_file,
        test_file = test_file,
        checkpoint_model = checkpoint_model,
        save_dir = checkpoint_save_dir,
        tensorboard_dir = tensorboard_save_dir,
        learning_rate = learning_rate,
        max_iter = max_iter,
        val_freq = val_freq,
        val_iter = val_iter)

