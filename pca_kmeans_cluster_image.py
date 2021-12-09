# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 00:50:05 2021

@author: user
"""

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import shutil


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#path = r"CHANGE TO DATASET LOCATION"
path = r"C:\Users\user\.spyder-py3\images_jkp"
#path = r"C:\Users\user\.spyder-py3\images_for_cluster"
path_ifc = "images_for_cluster/"
path_c = "images_clustered/"


# change the working directory to the path where the images are located
os.chdir(path)

# this list holds all the image filename
flowers = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        #if file.name.endswith('.png'):
        if file.name.endswith('.bmp'):
        #if file.name.endswith('.jpg'):
          # adds only the image files to the flowers list
            flowers.append(file.name)
print('after append filename...')
            
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

print('after VGG16...')

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}
#p = r"CHANGE TO A LOCATION TO SAVE FEATURE VECTORS"
p = r"C:\Users\user\.spyder-py3\feature_vector"

# lop through each image in the dataset
for flower in flowers:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(flower,model)
        data[flower] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)
print('after extract_features...')
 
# get a list of the filenames
filenames = np.array(list(data.keys()))
print('after filenames list...')
print('filenames = ', filenames)

# get a list of just the features
feat = np.array(list(data.values()))
print('after features list...')
print('feat = ', feat)

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)
print('after feature reshape...')
print('feat_reshape = ', feat)

# get the unique labels (from the flower_labels.csv)
#df = pd.read_csv('flower_labels.csv')
#label = df['label'].tolist()
#unique_labels = list(set(label))

# reduce the amount of dimensions in the feature vector
#pca = PCA(n_components=100, random_state=22)
pca = PCA(n_components=10, random_state=22)
pca.fit(feat)
x = pca.transform(feat)
print('after PCA feature reduce dimension...')
print('x = ', x)

# cluster feature vectors
#kmeans = KMeans(n_clusters=len(unique_labels),n_jobs=-1, random_state=22)
kmeans = KMeans(n_clusters=12, random_state=22)
kmeans.fit(x)
print('after kmeans clustering...')

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

'''
# copy clustered file in each gruop's folder 
if not os.path.isdir(path_c):
        os.mkdir(path_c)
'''
for i in range(len(groups)):
    print('\n')
    print(i, groups[i])
    path = str(i)
    #print(path)
    if not os.path.isdir(path):
        os.mkdir(path)
        
    for j in range(len(groups[i])):
        print(j, groups[i][j])
        shutil.copyfile(groups[i][j], path+"/"+groups[i][j])
#


print('after holds id and images...')
print('groups = ', groups)

# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        
#view_cluster(cluster)
#view_cluster(0)
#view_cluster(1)
#view_cluster(2)
#view_cluster(3)
        

# this is just incase you want to see which value for k might be the best 
sse = []
#list_k = list(range(3, 50))
list_k = list(range(4, 10))

print('begin different k value loop...')
for k in list_k:
    #km = KMeans(n_clusters=k, random_state=22, n_jobs=-1)
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)
    
    sse.append(km.inertia_)
    print("sse = ", sse)
print('after best k value...')


print('show plot...')
# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

print('Completed...')