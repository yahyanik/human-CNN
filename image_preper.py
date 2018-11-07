import caffe
import lmdb
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum

def write_images_to_lmdb(img_dir, db_name):
    for root, dirs, files in os.walk(img_dir, topdown = False):
        if root != img_dir:
            continue
        map_size = 300*300*3*5*len(files)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)
        for idx, name in enumerate(files):
            X = cv2.imread(os.path.join(root, name))
            y = 1
            datum = array_to_datum(X,y)
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())   
    txn.commit()
    env.close()
    print " ".join(["Writing to", db_name, "done!"])
    
    
write_images_to_lmdb('./training_set/my_set/val', 'val_set')