# Seyed Yahya Nikouei 
# Binghamton University ,NY
# Fall 2017
'''
mprof run <executable>
mprof plot
'''

#from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
#from keras.models import Sequential 
import imutils
import mxnet as mx
import time
import cv2

#model = Sequential()

#print cv2.__version__


#from guppy import hpy
#h = hpy()


txt1 = "./Models/Mob.prototxt"
txt1_1 = "Mobilenet.pyc"
txt2 = "./Models/deploy_goolgeNet.prototxt.txt"
model1 = "./Models/MobileNetSSD_deploy.caffemodel"
model2= "./Models/bvlc_googlenet.caffemodel"


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default = txt1,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default = model1,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.15,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

f3 = "./Test_Videos/EC-Main-Entrance-2017-05-21_14h20min25s670ms.mp4"
f1 = "./Test_Videos/EC-Main-Entrance-2017-05-21_02h10min05s000ms.mp4"
f4 = "./Test_Videos/EC-Main-Entrance-2017-05-21_15h05min30s000ms.mp4"
f2 = "./Test_Videos/vtest.avi"

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["person"]
COLORS = [0,255,0]

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (300,400))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
 
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=1).start()
vs = cv2.VideoCapture(f2)  ##################################################################### video
time.sleep(2.0) 
fps = FPS().start() 
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ok ,frame = vs.read()
    if not ok:
        print "nothing to read"
        
        
#    frame = imutils.resize(frame, width=400)
    frame1 = imutils.resize(frame, width=min((400, frame.shape[1])))
    frame = imutils.resize(frame, width=min((200, frame.shape[1])))
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame1.shape[:2]
    blob = cv2.dnn.blobFromImage(frame1, 0.007843, (300, 300), 127.5)
#    print blob.__sizeof__()
#    print blob
    
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    
#    print frame.shape[:2]
#    cv2.imshow("blob", blob)
    detections = net.forward()
  #  print "after"
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
#        print idx 
#        print confidence
 
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if detections[0, 0, i, 2] > args["confidence"] and detections[0, 0, i, 1] ==15 :
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
#            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
 
            # draw the prediction on the frame
#            label = "{}: {:.2f}%".format(CLASSES,
#                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
#            cv2.putText(frame, label, (startX, y),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)
    cv2.imshow("Frame", frame)
 #   cv2.imshow("Frame1", blob[0])
#    out.write(frame)
    key = cv2.waitKey(100) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
    # update the FPS counter
    fps.update()  
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
vs.release()
out.release()
cv2.destroyAllWindows()
vs.stop()        
            
            
            
            
            
            

            
            
            
            
            