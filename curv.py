from __future__ import division
import cv2
import numpy as np
import math 
import matplotlib.pyplot as plt
import random
from matplotlib import style
import xmltodict
import imutils
from sklearn import metrics
  
  
def fasele (grndtr, dir):
    dum= -1
    fnr = False
    fpr = False
    fasl = -1
    l1 = 0
    for d in dir :
        dum= (d[0] - grndtr[0])**2 + (d[1] - grndtr[1])**2
        if fasl == -1 or dum < fasl :
            fasl = dum
            l1= d
    if l1 == 0:
        fnr = True
    if l1 > 100 :
        fpr = True 
    return (fnr,fpr)
  
def function1 (img_name,grndtr):
    txt1 = "./Models/Mob.pyc"
    model1 = "./Models/MobileNetSSD_deploy.caffemodel"
    frame = cv2.imread(img_name)
    net = cv2.dnn.readNetFromCaffe("./Models/mine.prototxt", "./Models/MobileNetSSD_deploy.caffemodel")
    frame1 = imutils.resize(frame, width=min((400, frame.shape[1])))
    (h, w) = frame1.shape[:2]
    blob = cv2.dnn.blobFromImage(frame1, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    dir = []
    for i in np.arange(0, detections.shape[2]):
        idx, confidence = detections[0, 0, i, 1], detections[0, 0, i, 2]
        if confidence > 0.15 and idx ==15 :
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            dir.append(((endX+startX)/2, (endY+startY)/2))
      
    gg = fasele (grndtr,dir)
      
#             cv2.rectangle(frame, (startX, startY), (endX, endY),[0,255,0], 2)
    return gg
  
  
def function2 (img_name,grndtr):
    frame = cv2.imread(img_name)
    blurred = cv2.GaussianBlur(frame, (17, 17), 0)   #blur is useful to reduce the false positive and negatives
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    (faces, weights)= HOGCascade.detectMultiScale(gray ,winStride=(8,8),padding=(32,32),scale=1.05)
    dir = []
    for (x,y,w,h) in faces:
        dir.append((x+w/2,y+h/2))
    gg = fasele (grndtr,dir)
      
#             cv2.rectangle(frame, (startX, startY), (endX, endY),[0,255,0], 2)
    return gg
  
  
def function3 (img_name,grndtr):
    frame = cv2.imread(img_name)
    face_cascade = cv2.CascadeClassifier('haarcascade_fullbody_1.xml')
    blurred = cv2.GaussianBlur(frame, (17,17), 0)   #blur is useful to reduce the false positive and negatives
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    dir = []
    for (x,y,w,h) in faces:
        dir.append((x+w/2,y+h/2))
    gg = fasele (grndtr,dir)
      
#             cv2.rectangle(frame, (startX, startY), (endX, endY),[0,255,0], 2)
    return gg
  
  
# def function4 (img_name):
#     txt1 = "./Models/Mob.pyc"
#     model1 = "./Models/MobileNetSSD_deploy.caffemodel"
#     frame = cv2.imread(img_name)
#     net = cv2.dnn.readNetFromCaffe("./Models/Mob.pyc", "./Models/MobileNetSSD_deploy.caffemodel")
#     frame1 = imutils.resize(frame, width=min((400, frame.shape[1])))
#     (h, w) = frame1.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame1, 0.007843, (300, 300), 127.5)
#     net.setInput(blob)
#     detections = net.forward()
#     dir = []
#     for i in np.arange(0, detections.shape[2]):
#         idx, confidence = detections[0, 0, i, 1], detections[0, 0, i, 2]
#         if confidence > 0.15 and idx ==15 :
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             dir.append(((endX-startX)/2, (endY- startY)/2))
# #             cv2.rectangle(frame, (startX, startY), (endX, endY),[0,255,0], 2)
#     return dir
  
  
  
  
  
  
HOGCascade = cv2.HOGDescriptor()    
HOGCascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
k = 0

fnrnn = 0
fprnn = 0
fnrsvm = 0
fprsvm = 0
fnrcas = 0
fprcas = 0
y_test = []
file = open("./training_set/VOC2012/ImageSets/Main/person_val.txt", "r") #read each file
ff = file.readlines()
  
  
  
for i in ff:
    if i[12]  !='-' :
        name = './training_set/VOC2012/Annotations/'+i[:11]+'.xml'
#         print name
        dg = open(name, "r") #read each file
  
        f = dg.read()
        df = xmltodict.parse(f) #change format to dictionary
#         print df
#         print name
#         print df
        try :
            for j in df['annotation']['object']:
#             print j
#             print df['annotation']['object']
#             for per in j:
                grndtr =  ((int(j['bndbox']['xmin'])+int(j['bndbox']['xmin']))/2,(int(j['bndbox']['ymin'])+int(j['bndbox']['ymin']))/2)
                y_test.append(grndtr)
             
                if j['name'] =='person' :
                    k+=1
                    img_name = './training_set/VOC2012/JPEGImages/' + str(df['annotation'] ['filename'])
                    fnr, fpr = function1(img_name,grndtr)
                    if fnr :
                        fnrnn+=1
                    if fpr :
                        fprnn+=1
                    fnr, fpr =  function2(img_name,grndtr)
                    if fnr :
                        fnrsvm+=1
                    if fpr :
                        fprsvm+=1
                    fnr, fpr =  function3(img_name,grndtr)
                    if fnr :
                        fnrcas+=1
                    if fpr :
                        fprcas+=1
        except :
                if df['annotation']['object']['name'] == "person" :
                    k+=1
                    img_name = './training_set/VOC2012/JPEGImages/' + str(df['annotation'] ['filename'])
                    fnr, fpr =  function1(img_name,grndtr)
                    if fnr :
                        fnrnn+=1
                    if fpr :
                        fprnn+=1
                    fnr, fpr =  function2(img_name,grndtr)
                    if fnr :
                        fnrsvm+=1
                    if fpr :
                        fprsvm+=1
                    fnr, fpr =  function3(img_name,grndtr)
                    if fnr :
                        fnrcas+=1
                    if fpr :
                        fprcas+=1
    print k
    if k>= 2000 :
        break
    
fnrnn = fnrnn/k
fprnn = fprnn/k
fnrsvm = fnrsvm/k
fprsvm = fprsvm/k
fnrcas = fnrcas/k
fprcas = fprcas/k
               
               
print 'fnrnn :' , fnrnn
print 'fprnn :' , fprnn
print 'fnrsvm :' , fnrsvm
print 'fprsvm :' , fprsvm
print 'fnrcas :' , fnrcas
print 'fprcas :' , fprcas
      
# fpr, tpr, threshold = metrics.roc_curve(y_test, nn)                   
# roc_auc = metrics.auc(fpr, tpr)
#  
# # method I: plt
# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
                      
                      
  
  
# net = cv2.dnn.readNetFromCaffe("./Models/Mob.pyc", "./Models/MobileNetSSD_deploy.caffemodel")
# frame1 = imutils.resize(frame, width=min((400, frame.shape[1])))
# (h, w) = frame1.shape[:2]
# blob = cv2.dnn.blobFromImage(frame1, 0.007843, (300, 300), 127.5)
# net.setInput(blob)
# detections = net.forward()
# for i in np.arange(0, detections.shape[2]):
#         idx, confidence = detections[0, 0, i, 1], detections[0, 0, i, 2]
#         if confidence > 0.15 and idx ==15 :
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
# 
#             cv2.rectangle(frame, (startX, startY), (endX, endY),[0,255,0], 2)
# cv2.imshow("Frame", frame)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
# class XmlListConfig(list):
#     def __init__(self, aList):
#         for element in aList:
#             if element:
#                 # treat like dict
#                 if len(element) == 1 or element[0].tag != element[1].tag:
#                     self.append(XmlDictConfig(element))
#                 # treat like list
#                 elif element[0].tag == element[1].tag:
#                     self.append(XmlListConfig(element))
#             elif element.text:
#                 text = element.text.strip()
#                 if text:
#                     self.append(text)
# 
# class XmlDictConfig(dict):
#     '''
#     Example usage:
# 
#     >>> tree = ElementTree.parse('your_file.xml')
#     >>> root = tree.getroot()
#     >>> xmldict = XmlDictConfig(root)
# 
#     Or, if you want to use an XML string:
# 
#     >>> root = ElementTree.XML(xml_string)
#     >>> xmldict = XmlDictConfig(root)
# 
#     And then use xmldict for what it is... a dict.
#     '''
#     def __init__(self, parent_element):
#         if parent_element.items():
#             self.update(dict(parent_element.items()))
#         for element in parent_element:
#             if element:
#                 # treat like dict - we assume that if the first two tags
#                 # in a series are different, then they are all different.
#                 if len(element) == 1 or element[0].tag != element[1].tag:
#                     aDict = XmlDictConfig(element)
#                 # treat like list - we assume that if the first two tags
#                 # in a series are the same, then the rest are the same.
#                 else:
#                     # here, we put the list in dictionary; the key is the
#                     # tag name the list elements all share in common, and
#                     # the value is the list itself 
#                     aDict = {element[0].tag: XmlListConfig(element)}
#                 # if the tag has attributes, add those to the dict
#                 if element.items():
#                     aDict.update(dict(element.items()))
#                 self.update({element.tag: aDict})
#             # this assumes that if you've got an attribute in a tag,
#             # you won't be having any text. This may or may not be a 
#             # good idea -- time will tell. It works for the way we are
#             # currently doing XML configuration files...
#             elif element.items():
#                 self.update({element.tag: dict(element.items())})
#             # finally, if there are no child tags and no attributes, extract
#             # the text
#             else:
#                 self.update({element.tag: element.text})
# 
# 
# df = pd.read_csv('./training_set/VOC2012/Annotations/2009_002523.xml')
#     
# print df.head()
# 
# tree = ElementTree.parse('./training_set/VOC2012/Annotations/2009_002523.xml')
# root = tree.getroot()
# xmldict = XmlDictConfig(root)
# 
  
  
  
# with open ('./training_set/VOC2012/Annotations/2009_002523.xml','r') as f:
#     dk = f.readlines()
#     
# dr =loadtxt('./training_set/VOC2012/Annotations/2009_002523.xml')
# print dr
  
  
  
  
  
  
  
