# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:16:10 2021

@author: wmchiang
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib notebook
import matplotlib.pyplot as plt
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize

from joblib import dump, load
import pandas as pd


#path = "images_cat/"
path = "images_clustered/"

dim = 64
def load_image_files(container_path, dimension=(dim, dim)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            #img = skimage.io.imread(file)
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')            
            #print('img_resized.size = ', img_resized.size)
            flat_data.append(img_resized.flatten()) 
            print('flat_data.__len__() = ', flat_data.__len__())
            #print('flat_data.__len__()/img_resized.size = ', " ("+flat_data.__len__(), "/", img_resized.size+")")
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


print('Load image files with categories as subfolder names')
print('which performs like scikit-learn sample dataset...')
image_dataset = load_image_files(path)

print('image_dataset.data = ', image_dataset.data)
print('image_dataset.target = ', image_dataset.target)
print('image_dataset.target_names = ', image_dataset.target_names)
print('image_dataset.images = ', image_dataset.images)
print('image_dataset.DESCR = ', image_dataset.DESCR)


# generate file for SVM platform
print('Save dataset.data & dataset.target to file...')
df_d = pd.DataFrame(data=image_dataset['data'])
#print("type = ", type(df_d[0][0]))
float_formatter = "{:.4f}".format

for i in range(len(image_dataset.data)):    
    for j in range(dim*dim*3):
        print('(i, j) = ('+str(i)+','+str(j)+')')
        df_d[j][i] = float_formatter(df_d[j][i])

df_d.insert(dim*dim*3, "label", image_dataset.target)
print('df_d', df_d)


filename = 'cat_'+str(dim)+'x'+str(dim)+'.txt'
df_d.to_csv(filename, sep = ',', index = False)

file_data = ""
with open(filename, "r", encoding="utf-8") as f:
    isFirstLine = True
    for line in f:
        if (isFirstLine):
            isFirstLine = False
        else:    
            line = line[::-1].replace(',', ';', 1)[::-1]
            file_data += line
with open(filename, "w", encoding="utf-8") as f:
    f.write(file_data)
#input()


'''
print('Split arrays or matrices into random train and test subsets...')
# X = flat_data, y = target (label)
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)
print('X_train = ', X_train)
print('X_test = ', X_test)
print('y_train = ', y_train)
print('y_test = ', y_test)


print('Set SVM parameters...')
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

print('Run fit with all sets of parameters...')
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
#
#dump(clf, 'filename.joblib')
#clf = load('filename.joblib')
#
clf.fit(X_train, y_train)

print('Call predict on the estimator with the best found parameters...')
y_pred = clf.predict(X_test)

print('Build a text report showing the main classification metrics...')
print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))
'''