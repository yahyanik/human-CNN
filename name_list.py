import os
import csv

'''

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
#DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../data/DogsCatsKaggle/train'))
TXT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, './training_set/my_set'))


list1 = os.listdir('./training_set/my_set/train')

#resultFyle = open("trian.csv",'wb')
#wr = csv.writer(resultFyle, dialect='excel')
#for w1 in list1:
#    w = list(w1)
#    z = [''.join(w)]

    

#    wr.writerow(z)
 
with open('{}/train.txt'.format(TXT_DIR), 'w') as f:
    for image in list1:
        f.write('./{} 0\n'.format(image))
#    for image in cat_train:
#        f.write('{} 1\n'.format(image))
    f.close()    
    
list1 = os.listdir('./training_set/my_set/val')

with open('{}/val.txt'.format(TXT_DIR), 'w') as f:
    for image in list1:
        f.write('./{} 0\n'.format(image))
#    for image in cat_test:
#        f.write('{} 1\n'.format(image))
    f.close()
    
   
    
'''


import caffe  
ssd_net = caffe.Net('hu_deploy.prototxt', caffe.TEST) # or caffe.TRAIN  
ssd_net.forward()