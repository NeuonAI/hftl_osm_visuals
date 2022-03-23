# Neuon AI - PlantCLEF 2021

from six.moves import cPickle
import numpy as np
#
source = "PATH_TO_SAVED_PREDICTION_PKL_FILE" # .pkl


with open(source,'rb') as fid:
    pred_dict = cPickle.load(fid)
    
def get_rank(dict_value):
    prob = dict_value['prob']
    label = dict_value['label']
    
    idx = np.argsort(prob)[::-1]
    
    np.argmax(prob) == label
    
    rank_i = np.squeeze(np.where(idx==label)) + 1
    
    return rank_i

ranks = np.asarray([get_rank(value) for key,value in pred_dict.items()])
mrr = np.sum((1/ranks))/len(pred_dict)

