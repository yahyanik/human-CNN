

#from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
#from keras.models import Sequential 
import imutils
import time
import cv2



txt1 = "./Models/MobileNetSSD_deploy.prototxt.txt"
txt2 = "bvlc_googlenet.prototxt"
txt3 = "./Models/VGG.prototxt"
txt4 = "./Models/squeezenet_v1.0.prototxt"
model1 = "./Models/MobileNetSSD_deploy.caffemodel"
model2= "bvlc_googlenet.caffemodel"
model3= "./Models/VGG_ILSVRC_19_layers.caffemodel"
model4= "./Models/squeezenet_v1.0.caffemodel"


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default = txt4,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default = model4,
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

'''
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
        
        
    frame = imutils.resize(frame, width=400)
#    frame = imutils.resize(frame, width=min((400, frame.shape[1])))
 
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
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
        idx, confidence = detections[0, 0, i, 1], detections[0, 0, i, 2]
#        print idx 
#        print confidence
 
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"] and idx ==15 :
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
#            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
 
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES,
                confidence * 100)
            cv2.rectangle(frame, (startX-35, startY), (endX-35, endY),
                COLORS, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
#            cv2.putText(frame, label, (startX, y),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)
    cv2.imshow("Frame", frame)
 #   cv2.imshow("Frame1", blob[0])
#    out.write(frame)
    key = cv2.waitKey(10) & 0xFF
 
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
'''
rows = open('synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

fps = FPS().start()           
vs = cv2.VideoCapture(f2)
#image = cv2.imread('./Test_Videos/image2.jpg')   
while True:         
    ok ,frame = vs.read()     
    (h, w) = frame.shape[:2]       
    blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), (104, 117, 123)) 
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])    
    net.setInput(blob)
#start = time.time()
    preds = net.forward()
    idxs = np.argsort(preds[0])[::-1][:1]
#    for (i, idx) in enumerate(idxs):
#        idx, confidence = detections[0, 0, i, 1], detections[0, 0, i, 2]
        
#        if i == 0:
#            text = "Label: {}, {:.2f}%".format(classes[idx],
#                preds[0][idx] * 100)
#            cv2.putText(frame, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#                0.7, (0, 0, 255), 2)
 
    # display the predicted label + associated probability to the
    # console    
#        print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
#            classes[idx], preds[0][idx]))
        
        
        
        
        
        
#end = time.time()
#print("[INFO] classification took {:.5} seconds".format(end - start))








#cv2.imshow("Image", image)
    cv2.imshow("Image1", frame)
    key = cv2.waitKey(10) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break     
    fps.update()        
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))     

vs.release()
cv2.destroyAllWindows()
vs.stop()  
         