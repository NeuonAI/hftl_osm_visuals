# Comparing HFTL and OSM Networks in the context of Cross-Domain Plant Identification
This repository contains the implementation method of our Herbarium-Field Triplet Loss Network (HFTL Network) and One-streamed Mixed Network (OSM Network) in the context of Cross-Domain Plant Identification. Our results show that the HFTL Network can generalize rare species as equally as species with many training data better than the OSM Network (conventional CNNs). 

Figure A and B below show the Top-5 predictions of a plant sample with its predicted scores and activation maps from the HFTL and OSM Networks respectively. More samples of comparison can be found [here](https://drive.google.com/drive/folders/1QkzOlP8atWIriYAE6Tq2e53ZEjZONH5R?usp=sharing).
HFTL Network               |  OSM Network
:-------------------------:|:-------------------------:
![Figure 1](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/figures/HFTL_IR_prediction_1_19165_21526.jpg "HFTL Network Top-5 predictions")  |  ![Figure 2](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/figures/OSM_IR_prediction_1_19165_21526.jpg "OSM Network Top-5 predictions")


## Research article
A machine learning approach for cross-domain plant identification using herbarium specimens <br>
[https://doi.org/10.1007/s00521-022-07951-6](https://doi.org/10.1007/s00521-022-07951-6)
 
## Requirements
- Tensorflow 1.12
- [TensorFlow-Slim library](https://github.com/tensorflow/models/tree/r1.12.0/research/slim)
- [imagenet pretrained models (inception v4 and inception resnet v2)](https://github.com/tensorflow/models/tree/r1.12.0/research/slim#pre-trained-modelss)
<br /> 

## Data
- [PlantCLEF 2021](https://www.aicrowd.com/challenges/lifeclef-2021-plant)
- [PlantCLEF 2017](https://www.imageclef.org/lifeclef/2017/plant)

## Scripts
### Training scripts
- HFTL Network
  - [run_HFTL_network.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/run_HFTL_network.py)
  - [network_module_HFTL.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/network_module_HFTL.py) 
- OSM Network
  - [run_OSM_network.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/run_OSM_network.py)
  - [network_module_OSM.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/network_module_OSM.py)

### Validation scripts
- HFTL Network
  1. [get_herbarium_dictionary_HFTL.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/get_herbarium_dictionary_HFTL.py)
  2. [validate_HFTL_network.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/validate_HFTL_network.py)
  3. [get_mrr_score.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/get_mrr_score.py)
 
- OSM Network
  1. [get_herbarium_dictionary_OSM.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/get_herbarium_dictionary_OSM.py)
  2. [validate_OSM_network.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/validate_OSM_network.py)
  3. [get_mrr_score.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/get_mrr_score.py)

### Visualizing activation map scripts
- HFTL Network
  - [visualise_prediction_HFTL-IR.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/59b88c4379f56fc52dccd7421d23cfccc440d2c0/visualise_prediction_HFTL-IR.py)
- OSM Network
  - [visualise_prediction_OSM-IR.py](https://github.com/NeuonAI/hftl_osm_visuals/blob/59b88c4379f56fc52dccd7421d23cfccc440d2c0/visualise_prediction_OSM-IR.py)

## Lists
#### Herbarium Network
- [clef2021_herbarium_combined_train_306005.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/herbarium/clef2021_herbarium_combined_train_306005.txt)
- [clef2021_herbarium_combined_test_15221.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/herbarium/clef2021_herbarium_combined_test_15221.txt)

#### Field Network (2017)
- [clef2017_EOL_Web_train_server.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/field_2017/clef2017_EOL_Web_train_server.txt)
 - [clef2017_EOL_Web_test_server.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/field_2017/clef2017_EOL_Web_test_server.txt)

#### Field Network (2021)
- [clef2021_field_combined_train.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/field_2021/clef2021_field_combined_train.txt)
- [clef2021_field_combined_test.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/field_2021/clef2021_field_combined_test.txt)
 
#### HFTL Network
- [clef2021_shared_class_herbarium_train.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/c757a4d956aca59739d39fd60406a1e6c7a8b8a5/list/HFTL/clef2021_shared_class_herbarium_train.txt)
- [clef2021_shared_class_herbarium_test.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/c757a4d956aca59739d39fd60406a1e6c7a8b8a5/list/HFTL/clef2021_shared_class_herbarium_test.txt)
- [clef2021_shared_class_field_train.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/c757a4d956aca59739d39fd60406a1e6c7a8b8a5/list/HFTL/clef2021_shared_class_field_train.txt)
- [clef2021_shared_class_field_test.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/c757a4d956aca59739d39fd60406a1e6c7a8b8a5/list/HFTL/clef2021_shared_class_field_test.txt)

#### OSM Network
- [clef2021_mixed_class_train.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/OSM/clef2021_mixed_class_train.txt)
- [clef2021_mixed_class_test.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/OSM/clef2021_mixed_class_test.txt)

#### Herbarium Dictionary
- [herbarium_dictionary.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/herbarium_dictionary.txt)
- [herbarium_dictionary_with_field_bias.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/herbarium_dictionary_with_field_bias.txt)
 
#### Test Sets
- [test_set_1_seen.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/test_set_1_seen.txt) (with field training images)
- [test_set_2_unseen.txt](https://github.com/NeuonAI/hftl_osm_visuals/blob/a18b595d9e42c78156ed25c9c9b1124d2f14a4d0/list/test_set_2_unseen.txt) (without field training images) <br />
  (note: the [image files from Test Set 2](https://github.com/NeuonAI/hftl_osm_visuals/tree/main/planttest) are sourced from Google Image queries)

## Checkpoints / Trained models
- HFTL Network
  - [HFTL inception resnet v2 model](https://github.com/NeuonAI/hftl_osm_visuals/tree/main/checkpoints/HFTL/inception_resnet_v2)
  - [HFTL inception v4 model](https://github.com/NeuonAI/hftl_osm_visuals/tree/main/checkpoints/HFTL/inception_v4)
- OSM Network
  - [OSM inception resnet v2 model](https://github.com/NeuonAI/hftl_osm_visuals/tree/main/checkpoints/OSM/inception_resnet_v2)
  - [OSM inception v4 model](https://github.com/NeuonAI/hftl_osm_visuals/tree/main/checkpoints/OSM/inception_v4)

