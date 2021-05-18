import json

# with open('./data/annotations.json') as json_file:
# 	data = json.load(json_file)
# 	# print(data)

import csv
from collections import defaultdict
import numpy as np
import pickle
from  keras_utils import classes_yolo as classes 

classes_initial = ['glass','paper','cardboard','plastic','metal','trash']
# classes = ['Plastic bag & wrapper','Cigarette','Unlabeled litter','Bottle','Bottle cap',
# 			'Can','Other plastic','Carton','Cup','Straw'] # just the first 10 classes

labels = defaultdict(list)
with open('meta_df.csv', mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	line_count = 0
	for row in csv_reader : #4785 annotations
		if line_count >= 4784:
			print("t")
			break

		if line_count == 0:	
			line_count += 1
			continue

		x = float(row["x"])
		y = float(row["y"])
		w = float(row["width"])
		h = float(row["height"])
		im_w = float(row["img_width"])
		im_h = float(row["img_height"])
		c_id = row["supercategory"]
		
		if x < 0:
			x = 0
		if y < 0:
			y = 0
		
		# if x > im_w:
		# 	x = im_w
		# if y > im_w:
		# 	y = 0					
		if c_id not in classes:
			continue

		c_tensor = np.zeros(len(classes))

		c_tensor[classes.index(c_id)] = 1
		an = (((im_w,im_h),(x,y),(float(row["x"])+w,float(row["y"])+h)),c_tensor)

		labels[row["img_file"]].append(an)
		
		line_count += 1
	# print(f'Processed {line_count} lines.')


l1 = dict(list(labels.items())[59*len(labels)//60:])
l2 = dict(list(labels.items())[:-len(labels)//60])


print(len(l1.keys()))
print(len(l2.keys()))
f = open("annotations_test.pkl","wb")
pickle.dump(l1,f)
f.close()

f = open("annotations.pkl","wb")
pickle.dump(l2,f)
f.close()

f = open("annotations_full.pkl","wb")
pickle.dump(labels,f)
f.close()

print(len(labels.keys()))

# print(labels["batch_10/000011.jpg"])
# print(data['annotations'][0]['image_id'])
# print(data['annotations'][0])