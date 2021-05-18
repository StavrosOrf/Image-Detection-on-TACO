import json
import taco
from random import random

final_data = {}
with open('./dataset/annotations.json') as json_file:
	data = json.load(json_file)

# with open('via_region_data.json') as json_file:
# 	data2 = json.load(json_file)	
# print(final_data['batch_1/000006.jpg']) 
filenames = data['images']

cat_ids = {}
super_cat_ids = {}
for i,a in enumerate(data['categories']):
	cat_ids[i] = a['name']
	super_cat_ids[i] = a['supercategory']

image_ids = {}
for i,a in enumerate(data['images']):
	# print(a['supercategory'])
	# if a['supercategory'] not in taco.classes or a['supercategory'] in cat_ids:
	# 	continue
	size = a['height'] * a['width']
	final_data[a['file_name']] = {'filename': a['file_name'],
							 		'size' : size,'regions':[], 'file_attributes':
							 		{'caption': '', 'public_domain': 'no', 'image_url': ''}
							 		} 
	image_ids[i] = a['file_name']


classes_yolo = ['Plastic bag & wrapper','Paper','Unlabeled litter','Bottle','Bottle cap',
			'Can','Other plastic','Carton','Cup','Straw'] 

for a in data['annotations']:

	img_name = image_ids[a['image_id']]
	# category = cat_ids[a['category_id']]
	category = super_cat_ids[a['category_id']]
	if category not in classes_yolo:
		print("skip")
		continue

	#maybe skip some categories
	x = []
	y = []
	flag = True
	for s in a['segmentation'][0]:		
		if flag:
			x.append(s)
		else:
			y.append(s)			
		flag = not flag


	t = {'shape_attributes':{'name':'polygon','all_points_x':x,'all_points_y':y},
		'region_attributes':{'name':category,'type':'unknown','image_quality':{
		'good':'true','frontal':'true','good_illumination':'true'}}}
	
	final_data[img_name]['regions'].append(t)		



# for i in cat_ids.keys():
# 	print("'" + cat_ids[i] + "',",end='')

# print(len(cat_ids.keys()))

supercategories = {}
for k in super_cat_ids.keys():
	# if 	super_cat_ids[i]
	supercategories[super_cat_ids[k]] = 1

for i in supercategories.keys():
	print("'" + str(i) + "',",end='')

print(len(supercategories.keys()))	




print("\n\n")

val = {}
train = {}
test = {}

for i in final_data.keys():
	if len(final_data[i]['regions']) == 0:		
		continue  

	value = random()
	if value > 0.8:
		val[i] = final_data[i]
	elif value > 0.78:
		test[i] = final_data[i]["filename"]		
	else:
		train[i] = final_data[i]		



# shuffle to two random sets of validation and train

print(len(val.keys()))
print(len(test.keys()))
print(len(train.keys()))

with open('./dataset/train/via_region_data.json', 'w') as outfile:
    json.dump(val, outfile)

with open('./dataset/val/via_region_data.json', 'w') as outfile:
    json.dump(train, outfile)

with open('test_images.json', 'w') as outfile:
    json.dump(test, outfile)    