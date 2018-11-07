
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tensorflow as tf

#from datasets import dataset_utils




base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/'
synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url)
filename, _ = urllib.request.urlretrieve(synset_url)
synset_list = [s.strip() for s in open(filename).readlines()]
num_synsets_in_ilsvrc = len(synset_list)
#assert num_synsets_in_ilsvrc == 1000
filename, _ = urllib.request.urlretrieve(synset_to_human_url)
synset_to_human_list = open(filename).readlines()
num_synsets_in_all_imagenet = len(synset_to_human_list)
#assert num_synsets_in_all_imagenet == 21842
synset_to_human = {}
for s in synset_to_human_list:
    parts = s.strip().split('\t')
    print (parts)
#    assert len(parts) == 2
    synset = parts[0]
    human = parts[1]
    synset_to_human[synset] = human

label_index = 1
labels_to_names = {0: 'background'}
for synset in synset_list:
    name = synset_to_human[synset]
    labels_to_names[label_index] = name
    label_index += 1
    
    
print (labels_to_names)