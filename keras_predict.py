import sys
import numpy as np
import cv2
import keras_utils as utils
import matplotlib.pyplot as plt
from keras import models
from keras.models import load_model
from keras_train import create_model
import random
import pickle

# if len(sys.argv) < 2:
#     print("Image name missing; \n usage: python predict.py <image name>")
#     sys.exit()
# img_name = sys.argv[1]


with open('annotations_test.pkl', 'rb') as f:
	label_frames = pickle.load(f)

label_keys = list(label_frames.keys())
print(label_keys)
exit(0)
suffix= "./data/"

weights = './models/yolo.h5'
weights = './models/run3v2__best.h5'
model = create_model(load_weights=True,include_yolo = True)
model.build((None,224,224,3))
model.load_weights(weights,by_name=True,skip_mismatch=True)

weights2 = './models/run2v2__best.h5'
model2 = create_model(load_weights=True,include_yolo = True)
model2.build((None,224,224,3))
model2.load_weights(weights2,by_name=True,skip_mismatch=True)

for i in range(15):
	# l = random.choice(label_keys)
	l = label_keys[i]
	img_name = suffix + l
		
	dims = (label_frames[l][0][0][0])
	boxes = []
	for d in label_frames[l]:
		boxes.append(((int(d[0][1][0]),int(d[0][1][1])),(int(d[0][2][0]),int(d[0][2][1])),1))	
	
	img = cv2.imread(img_name)
	img_float = cv2.resize(img, (224,224)).astype(np.float32)
	img_float -= 128

	draw1 = utils.draw_boxes(img, boxes, color=(0, 255, 255), thick=20, draw_dot=True, radius=3)
	draw1 = draw1.astype(np.uint8)
	

	img_in = np.expand_dims(img_float, axis=0)
	print(img_name)

	pred = model.predict(img_in)
	bboxes = utils.get_boxes(pred[0], cutoff=0.1,dims=dims)
	bboxes = utils.nonmax_suppression(bboxes, iou_cutoff = 0.2)
	draw = utils.draw_boxes(img, bboxes, color=(0, 0, 255), thick=20, draw_dot=True, radius=3)
	draw = draw.astype(np.uint8)
	print(bboxes)
	print(type(bboxes))

	pred = model2.predict(img_in)
	bboxes = utils.get_boxes(pred[0], cutoff=0.1,dims=dims)
	bboxes = utils.nonmax_suppression(bboxes, iou_cutoff = 0.2)
	draw2 = utils.draw_boxes(img, bboxes, color=(0, 0, 255), thick=20, draw_dot=True, radius=3)
	draw2 = draw2.astype(np.uint8)
	print(bboxes)
	print(type(bboxes))


	plt.subplot(131)
	plt.title('Ground-Truth')
	plt.imshow(draw1[...,::-1])
	plt.subplot(132)
	# plt.axis('off')
	plt.yticks([])
	plt.title('TrashNet Pretrained')
	plt.imshow(draw[...,::-1])
	plt.subplot(133)
	plt.yticks([])
	plt.title('Imagenet Pretrained')
	plt.imshow(draw2[...,::-1])
	plt.show()
