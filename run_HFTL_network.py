# Neuon AI - PlantCLEF 2021

from network_module_HFTL import network_module as net_module


image_dir_parent_train = "PATH_TO_PlantCLEF2020TrainingData"
image_dir_parent_test = "PATH_TO_PlantCLEF2020TrainingData"

train_file1 = "list/HFTL/clef2020_known_classes_herbarium_train_added.txt"
test_file1 = "list/HFTL/clef2020_known_classes_herbarium_test.txt"

train_file2 = "list/HFTL/clef2020_known_classes_field_train_added.txt"
test_file2 = "list/HFTL/clef2020_known_classes_field_test.txt"

checkpoint_model1 = "PATH_TO_PRETRAINED_HERBARIUM_CHECKPOINT_MODEL" # pretrained herbarium network plantclef 2021 .ckpt
checkpoint_model2 = "PATH_TO_PRETRAINED_FIELD_CHECKPOINT_MODEL" # pretrained field network plantclef 2021 / 2017 .ckpt

checkpoint_save_dir = "PATH_TO_CHECKPOINT_SAVE_DIR"


batch = 16
input_size = (299,299,3)
numclasses1 = 997
numclasses2 = 10000
learning_rate = 0.0001
iterbatch = 1
max_iter = 500000
val_freq = 60
val_iter = 20



network = net_module(
        batch = batch,
        iterbatch = iterbatch,
        numclasses1 = numclasses1,
        numclasses2 = numclasses2,
        input_size = input_size,
        image_dir_parent_train = image_dir_parent_train,
        image_dir_parent_test = image_dir_parent_test,
        train_file1 = train_file1,
        train_file2 = train_file2,
        test_file1 = test_file1,
        test_file2 = test_file2,        
        checkpoint_model1 = checkpoint_model1,
        checkpoint_model2 = checkpoint_model2,
        save_dir = checkpoint_save_dir,
        learning_rate = learning_rate,
        max_iter = max_iter,
        val_freq = val_freq,
        val_iter = val_iter
        )




