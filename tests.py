from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import numpy as np
import argparse
import json
from keras.utils.data_utils import get_file
import cv2


CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#    help="path to the input image")
args = vars(ap.parse_args())
 
# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = cv2.imread('./Test_Videos/image.jpg')

print("[INFO] loading and preprocessing image...")
image = image_utils.load_img("./Test_Videos/image.jpg", target_size=(224, 224))
image = image_utils.img_to_array(image)

image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet")
 
# classify the image
print("[INFO] classifying image...")
preds = model.predict(image)
#(inID, label) = decode_predictions(preds)
CLASS_INDEX = None
fpath = get_file('imagenet_class_index.json',CLASS_INDEX_PATH,cache_subdir='models')
CLASS_INDEX = json.load(open(fpath))
results = []
top = 1
for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)

(ZZ,inID, label) = (results[0])[0]
 
# display the predictions to our screen
print("ImageNet ID: {}, Label: {}".format(inID, label))
cv2.putText(orig, "Label: {}".format(label), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)

P = decode_predictions(preds)
(imagenetID, label, prob) = P[0][0]