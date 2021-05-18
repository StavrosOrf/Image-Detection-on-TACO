import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import cv2
from scipy.special import expit as sigmoid
from matplotlib import pyplot
import sys
import numpy as np
import warnings
import glob
import re
import pickle
import keras_utils as ku
from keras_utils import ResNet50
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
import datetime

import keras
from keras_applications import inception_v3 as inc_net
from keras_preprocessing import image
from keras_applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

CLASSES = 6
BATCH_SIZE = 32

# input = (224,224)

### Define generator and Import dataset (do test/train split)
def generator(label_keys, label_frames, batch_size=BATCH_SIZE, folder='data/'):
	'''
	Generator function
	# Arguments
	label_keys: image names, that are keys of the label_frames Arguments
	label_frames: array of frames (rows corresponding to a single image in the labels.csv)
	batch_size: batch size
	'''
	num_samples = len(label_keys)
	indx = label_keys

	while 1:
		shuffle(indx)
		for offset in range(0, num_samples, batch_size):
			batch_samples = indx[offset:offset+batch_size]

			images = []
			gt = []
			for batch_sample in batch_samples:
				# print(batch_sample)
				# print(label_frames[batch_sample])
				im = cv2.imread(folder+batch_sample)

				im = cv2.resize(im, (224,224))
				# im, frame = augument_data(batch_sample, label_frames[batch_sample])
				im = im.astype(np.float32)
				im -= 128
				images.append(im)				
				frame_tensor = label_to_tensor(label_frames[batch_sample])				
				gt.append(frame_tensor)

			X_train = np.array(images)
			# X_train =  K.applications.resnet50.preprocess_input(X_train)
			y_train = np.array(gt)
			yield shuffle(X_train, y_train)

def label_to_tensor(frame, imgsize=(224, 224), gridsize=(11,11), classes=len(ku.classes_yolo), bboxes=2):
	'''
	This function takes in the frame (rows corresponding to a single image in the labels.csv)
	and converts it into the format our network expects (coord conversion and normalization)
	'''
	grid = np.zeros(gridsize)

	y_span = imgsize[0]/gridsize[0]
	x_span = imgsize[1]/gridsize[1]

	class_prob = np.zeros((gridsize[0], gridsize[1], classes))
	confidence = np.zeros((gridsize[0], gridsize[1], bboxes))
	dims = np.zeros((gridsize[0], gridsize[1], bboxes, 4))

	for box in frame:
		c = np.zeros(classes)		
		((im_w,im_h),(x1,y1),(x2,y2)),(c) = box

		mult_w = im_w / imgsize[0]
		mult_h = im_h / imgsize[1]

		x1 = x1 // mult_w
		x2 = x2 // mult_w
		y1 = y1 // mult_h
		y2 = y2 // mult_h
					
		x_grid = int(((x1+x2)/2)//x_span)
		y_grid = int(((y1+y2)/2)//y_span)
		# if x2 >= 224:
		# 	print(x2)
		# 	x2 = 223
		# if y2 >= 224:
		# 	print(y2)
		# 	y2 = 223
		class_prob[y_grid, x_grid] = tuple(c)

		x_center = ((x1+x2)/2)
		y_center = ((y1+y2)/2)

		x_center_norm = (x_center-x_grid*x_span)/(x_span)
		y_center_norm = (y_center-y_grid*y_span)/(y_span)

		w = x2-x1
		h = y2-y1

		w_norm = w/imgsize[1]
		h_norm = h/imgsize[0]

		dims[y_grid, x_grid, :, :] = (x_center_norm, y_center_norm, w_norm, h_norm)

		grid[y_grid, x_grid] += 1

	tensor = np.concatenate((class_prob.ravel(), confidence.ravel(), dims.ravel()))
	return tensor  

def load_data1(label_keys, label_frames, batch_size=BATCH_SIZE, folder='data/'):
	'''
	Generator function
	# Arguments
	label_keys: image names, that are keys of the label_frames Arguments
	label_frames: array of frames (rows corresponding to a single image in the labels.csv)
	batch_size: batch size
	'''
	num_samples = len(label_keys)
	indx = label_keys

	shuffle(indx)

	batch_samples = indx[0:num_samples]

	images = []
	gt = []
	for i,batch_sample in enumerate(batch_samples):
		print(i)
		# print(batch_sample)
		# print(label_frames[batch_sample])
		# im = cv2.imread(folder+batch_sample)
		# im = cv2.resize(im, (224,224))
		frames= []
		for box in label_frames[batch_sample]:
			c = np.zeros(len(ku.classes))		
			((im_w,im_h),(x1,y1),(x2,y2)),(c) = box

			mult_w = im_w / 224
			mult_h = im_h / 224

			x1 = x1 // mult_w
			x2 = x2 // mult_w
			y1 = y1 // mult_h
			y2 = y2 // mult_h
			frames.append((((x1,y1),(x2,y2)),(c)))

		
		im, frame = ku.augument_data(batch_sample, frames,folder=folder)
		# im, frame = augument_data(batch_sample, label_frames[batch_sample])
		im = im.astype(np.float32)
		im -= 128
		images.append(im)				
		frame_tensor = ku.label_to_tensor(frame)				
		gt.append(frame_tensor)

	X_train = np.array(images)
	# X_train =  K.applications.resnet50.preprocess_input(X_train)
	y_train = np.array(gt)
	return shuffle(X_train, y_train)


def load_data(label_keys_frames, folder='GarbageClassificationImages/'):

	label_keys = []
	label_frames = []

	for l in label_keys_frames:
		s = l.split()
		label_keys.append(s[0])
		label_frames.append(s[1])

	num_samples = len(label_keys_frames)

	indx = label_keys


	shuffle(indx,label_frames)

	images = []
	gt = []
	batch_samples = indx[0:num_samples]
	for i,batch_sample in enumerate(batch_samples):

		img = cv2.imread(folder+batch_sample)
		if img is None:
			continue
		img = cv2.resize(img, (224,224))			
		images.append(img)                
		gt.append(int(label_frames[i])-1)

	X_train = np.array(images)
	y_train = np.array(gt)            
	X_train = K.applications.resnet50.preprocess_input(X_train)
	y_train = tf.keras.utils.to_categorical(y_train,CLASSES)
	print((X_train.shape,y_train.shape))
	return shuffle(X_train, y_train)

def create_model(load_weights,include_yolo,freeze_layers=False):

	res_model = K.applications.ResNet50(include_top = False, 
		weights ="imagenet",input_shape = (224,224,3),pooling="avg")

	if freeze_layers:
		for layer in res_model.layers[:143:]:
			layer.trainable = False

	model = K.models.Sequential(name="seq_layer")
	if not include_yolo:
		model.add(K.layers.Lambda(lambda image: tf.image.resize(image,((224,224)))))
	
	model.add(res_model)
	

	# model.add(K.layers.BatchNormalization())
	# model.add(K.layers.Dense(256,activation='relu'))
	# model.add(K.layers.Dropout(0.5))
	# model.add(K.layers.BatchNormalization())
	# model.add(K.layers.Dense(128,activation='relu'))
	# model.add(K.layers.Dropout(0.5))
	# model.add(K.layers.BatchNormalization())
	# model.add(K.layers.Dense(64,activation='relu'))
	# model.add(K.layers.Dropout(0.5))

	if not include_yolo:
		model.add(K.layers.Flatten(name="flat"))
		model.add(K.layers.Dense(CLASSES, activation='softmax',name="classifier_layer"))

	else:
		model.add(K.layers.Flatten(name="yolo_flat"))
		model.add(K.layers.Dense(2048,activation='relu',name="yolo_dns1"))
		model.add(K.layers.Dropout(0.5,name="yolo_drp1"))
		 # output tensor :
		# SS: Grid cells: 11*11
		# B: Bounding box per grid cell: 2
		# C: classes: len(ku.classes_yolo) #3,10,...
		# Coords: x, y, w, h per box: 4
		# tensor length: SS * (C +B(5) ) : 363--242--968 => 1573
		# tensor length: SS * (C +B(5) ) : 2420
		c = len(ku.classes_yolo)
		model.add(K.layers.Dense(11*11*(c+2*5),activation='linear',name="yolo_dns2"))

	# if load_weights:
	# 	# model.load_weights(model_name).expect_partial()
	# 	weights_list = load_weights(model_name)
		
	# 	for i, weights in enumerate(weights_list):
	# 		if i == 1:
	# 			print(weights)
	# 			# model.layers[i].set_weights(weights)
	# exit(0) 		
	# for i,layer in enumerate(model.layers):
	# 	print(i,layer.name,"-",layer.trainable)
	# 	if i == 1:
	# 		for j,l in enumerate(layer.layers):
	# 			if j>= 169:
	# 				# print(j,len(l.get_weights()))
	# 				print(j,np.sum(l.get_weights()))

	return model

def train_initial_resenet50(epochs,model_name):
	labels = []
	with open('one-indexed-files.txt') as f:
		lines = f.readlines()
		for l in lines:
			labels.append(l.rstrip("\n"))

	label_keys = list(labels)
	lbl_train, lbl_validn = train_test_split(label_keys, test_size=0.2)    

	train_x,train_y = load_data(lbl_train)
	test_x,test_y= load_data(lbl_validn)

	# model = create_model(load_weights=False,include_yolo = False)
	# model.build((None,224,224,3))
	# model.summary()

	model = ku.ResNet50(include_top=False, input_shape=(224,224,3),
                    load_weight=True, weights= 'imagenet')

	# # print(len(model.layers))
	# for layer in model.layers[:143:]:
	# 		layer.trainable = False

	# check_point = K.callbacks.ModelCheckpoint(filepath=model_name,monitor="val_loss",mode="auto",save_best_only="true",save_weights_only="true")
	check_point = ModelCheckpoint(filepath=model_name, monitor='val_loss', save_best_only=True, mode='auto', save_weights_only=True, verbose=1)
	log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"initResNet50ImagenetNoImagenetAVGPooling"
	# check_point = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)
	# model.compile(loss="categorical_crossentropy",optimizer=K.optimizers.RMSprop(lr=2e-5),metrics=['accuracy'])
	
	model.compile(loss="categorical_crossentropy",optimizer= RMSprop(lr=2e-5),metrics=['accuracy'])
	# model.compile(loss="categorical_crossentropy",optimizer= SGD(lr=0.001),metrics=['accuracy'])
	# model.summary()
	history = model.fit(train_x,train_y,batch_size=BATCH_SIZE,epochs=epochs,
		verbose=1,validation_data=(test_x,test_y),callbacks=[check_point])

	
	# model.save("./models/model_final.h5")
	model.save_weights("./models/last_model_resnet50.h5")

	pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='val')
	pyplot.legend()
	# plot accuracy during training
	pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='val')
	pyplot.legend()
	pyplot.show()

	# test_accuracy_resnet50(filepath = "./GarbageClassificationImagesTest/*.jpg", model_l = model)



def test_accuracy_resnet50(filepath,model_name=None, model_l = None):	
	if model_l is not None:
		model = model_l
	else:		
		# weights = './models/model_final.h5'
		# model = load_model(weights)
		model = create_model(load_weights=False,include_yolo = False)
		model = ku.ResNet50(include_top=False, input_shape=(224,224,3),
                    load_weight=True, weights= 'imagenet')
		print("created model")
		model.build((None,224,224,3))
		model.load_weights(model_name)
		# model.layers[1].summary()		
		# model = copy_model(model_name)
		print("loaded model")
	total = 0
	correct = 0
	confusion_matrix = np.zeros(shape=(len(ku.classes),len(ku.classes)))
	for img_name in glob.glob(filepath):
		img = cv2.imread(img_name)
		img = cv2.resize(img,(224,224))
		img = np.array(img)        
		img = K.applications.resnet50.preprocess_input(img)    
		img = np.expand_dims(img, axis=0)

		pred = model.predict(img)

		name = img_name.split('\\')[1]
		temp = re.compile("([a-zA-Z]+)([0-9]+)")
		res = temp.match(name).groups()
		name = res[0]
		# print("IMG:" + img_name.split('\\')[1] + "\t->\t" + ku.classes[np.argmax(pred)])
		if name == ku.classes[np.argmax(pred)]:
			correct = correct + 1

		confusion_matrix[ku.classes.index(ku.classes[np.argmax(pred)])][ku.classes.index(name)] += 1
		total += 1

		# explainer = lime_image.LimeImageExplainer()
		# explanation = explainer.explain_instance(img.astype('double'), model.predict(img), top_labels=6, hide_color=0, num_samples=6)
		
		# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=6, hide_rest=True)
		# plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

		# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=6, hide_rest=False)
		# plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

	print("\tTotal Images: " + str(total) + "\tAccuracy: " + str(correct*1.0/total))
	ku.plot_confusion_matrix(cm=confusion_matrix,target_names =ku.classes)
	

def train_resenetYolo_model(epochs,weights_path,model_name):	

	model = create_model(load_weights=True,include_yolo = True)
	model.build((None,224,224,3))
	# model.load_weights(weights_path,by_name=True,skip_mismatch=True)
	# model = ku.ResNet50(include_top=False, input_shape=(224,224,3),
 #                    load_weight=False, weights= None)
	print("created model")
	
	with open('annotations.pkl', 'rb') as f:
		label_frames = pickle.load(f)

	label_keys = list(label_frames.keys())
	lbl_train, lbl_validn = train_test_split(label_keys, test_size=0.2)

	train_x,train_y = load_data1(lbl_train,label_frames)
	test_x,test_y= load_data1(lbl_validn,label_frames)


	# train_generator = generator(lbl_train, label_frames)
	# validation_generator = generator(lbl_validn, label_frames)
	save_prefix = 'run2v2__best'

	# print(np.shape(next(train_generator)))	

	# check_point = K.callbacks.ModelCheckpoint(filepath=model_name,monitor="val_loss",mode="auto",save_best_only="true",save_weights_only="true")
	# model.compile(loss=ku.custom_loss,optimizer=K.optimizers.Adam(lr=0.001))
	model.compile(loss=ku.custom_loss,optimizer=K.optimizers.Adam(lr=0.001))
	model.summary()
	# optimizer = Adam(lr=0.001)
	# model.compile(optimizer=optimizer, loss=custom_loss)
	# '_weights.{epoch:02d}-{val_loss:.2f}
	model_checkpoint = ModelCheckpoint(filepath='models/' + save_prefix + ".h5", monitor='val_loss', save_best_only=True, mode='auto', save_weights_only=True, verbose=1)
	# history = model.fit_generator(train_generator, validation_data=validation_generator,
	# 								steps_per_epoch=len(lbl_train)//BATCH_SIZE, epochs=epochs,
	# 								validation_steps=len(lbl_validn)//BATCH_SIZE,
	# 								callbacks=[model_checkpoint])
	history = model.fit(train_x,train_y,batch_size=BATCH_SIZE,epochs=epochs,
		verbose=1,validation_data=(test_x,test_y),callbacks=[model_checkpoint])

	model.save("./models/yolo.h5")
	# model.save_weights("./models/last_model_resnet50")
	# history = history[5:]
	# pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	# plot accuracy during training
	# pyplot.subplot(212)
	# pyplot.title('Accuracy')
	# pyplot.plot(history.history['accuracy'], label='train')
	# pyplot.plot(history.history['val_accuracy'], label='test')
	# pyplot.legend()
	pyplot.show()
	return	
	
	# weights = './models/model_final.h5'
	# model = load_model(weights)
	# model = create_model(load_weights=False)
	# print("created model")
	# model.load_weights(model_name).expect_partial()

	# print("loaded model")
	filepath = "./GarbageClassificationImagesTest/*.jpg"
	total = 0
	correct = 0
	for img_name in glob.glob(filepath):
		img = cv2.imread(img_name)
		img = cv2.resize(img, (224,224))
		img = np.array(img)        
		img = K.applications.resnet50.preprocess_input(img)    
		img = np.expand_dims(img, axis=0)

		pred = model.predict(img)

		name = img_name.split('\\')[1]
		temp = re.compile("([a-zA-Z]+)([0-9]+)")
		res = temp.match(name).groups()
		name = res[0]
		# print("IMG:" + img_name.split('\\')[1] + "\t->\t" + classes[np.argmax(pred)])
		if name == ku.classes[np.argmax(pred)]:
			correct = correct + 1

		total = total + 1
	print("\tTotal Images: " + str(total) + "\tAccuracy: " + str(correct*1.0/total))

# =======================================================================================
# =======================================================================================
# ===================================== MAIN ==========================================
# =======================================================================================
# =======================================================================================
if __name__ == "__main__":
	model_name = "./models/best_model_resnet50_v5_2.h5";
	model_name = "./models/best_model_resnet50_v6.h5";
	model_name_yolo = "./models/best_model_yolo";
	train_initial_resenet50(epochs = 15,model_name = model_name)
	# model_name = "./models/yolo.h5"

	

	# model = create_model(load_weights=True,include_yolo = True)
	# model.build((None,224,224,3))
	# model.load_weights(model_name,by_name=True,skip_mismatch=True)
	# model_name = "./models/last_model_resnet50.h5"
	test_accuracy_resnet50(filepath = "./GarbageClassificationImagesTest/*.jpg", model_name = model_name)

	# train_resenetYolo_model(epochs = 50,weights_path=model_name,model_name = model_name_yolo)

