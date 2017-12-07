import numpy as np
import os
import tensorflow as tf
import cv2
import re
import time

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

# What model to download.
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_label(file):
	with open(file,'r') as fi:
		match = re.finditer('item {\n  name: \"(?P<name>.*)\"\n  id: (?P<id>\d+)\n  display_name: \"(?P<label>.*)\"\n}',fi.read())
	return [m.groupdict() for m in match]

# get all test images
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

labels = load_label(PATH_TO_LABELS)
colors = np.random.uniform(0, 255, size=(len(labels)+1, 3))

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    for image_path in TEST_IMAGE_PATHS:
      # using cv2 handle image
      image = cv2.imread(image_path)
      (h, w) = image.shape[:2]
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image, axis=0)

      # Actual detection.
      t = time.time()
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      print("Runtime:", time.time()-t)

      # Visualization of the results of a detection.
      for i in range(0,int(num[0])):
        confidence = scores[0,i]
        if confidence>0.5:
            idx = int(classes[0,i])
            box = boxes[0, i, 0:4] * np.array([h, w, h, w])
            (y, x, y2, x2) = box.astype("int")
            # draw rect
            cv2.rectangle(image, (x, y), (x2, y2), colors[idx], 2)
            # draw label
            label = "{}: {:.2f}%".format([t['label'] for t in labels if t['id'] == str(idx)][0], confidence * 100)
            print("{}".format(label))
            (fontX, fontY) = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 1)[0]
            y = y + fontY if y-fontY<0 else y
            cv2.rectangle(image,(x, y-fontY),(x+fontX, y),colors[idx],cv2.FILLED)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
      cv2.imshow(image_path,image)

cv2.waitKey(0)