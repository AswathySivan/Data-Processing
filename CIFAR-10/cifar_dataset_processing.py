import os
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

input_shape=[224,224,3]

#folder details
#dataset_folder="cifar"
dataset_folder="path to dataset folder"
train_folder_name=os.path.join(dataset_folder,"train")
test_folder_name=os.path.join(dataset_folder,"test")
labels_list=os.path.join(dataset_folder,"labels.txt")

train_folder=os.listdir(train_folder_name)

f=open(labels_list,"r") 
f=f.readlines()
labels_names=[]
for line in f:
	labels_names.append(line.rstrip())

train_words=[]
for train_files in train_folder:
	words=train_files.split("_")
	words=words[1].split(".")
	train_words.append(words[0])

#Appending images and labels in to list
train_data=[]
train_labels=[]
for k in range(len(labels_names)):
	indices = [i for i, x in enumerate(train_words) if x == labels_names[k]]
	for index1 in indices:
			train_labels.append(k)
			img_name=train_folder[int(index1)]
			img=cv2.imread(os.path.join(train_folder_name,img_name))

			img = np.array(img)
			img = np.resize(img, (input_shape[0], input_shape[1], input_shape[2]))
			img = img.reshape( input_shape[0], input_shape[1], input_shape[2])
			img = img.astype("float32") / 255.0
			train_data.append(img)

dfObj = pd.DataFrame(train_labels) 	
onehotencoder = OneHotEncoder(categorical_features = [0]) 
dfObj = onehotencoder.fit_transform(dfObj).toarray() 
train_label = dfObj

#Generate seperate files for  images and labels
np.save("Train_data.npy",train_data)
np.save("Train_label.npy",train_label)
