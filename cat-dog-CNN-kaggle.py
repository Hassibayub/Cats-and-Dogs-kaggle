import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
import os 
import pickle

CATEGORIES = {
    'cat':0,
    'dog':1
}

def FIND_TYPE(val):
    if 'dog' in root:
        return (CATEGORIES['dog'])
    elif 'cat' in root:
        return (CATEGORIES['cat'])

    
def IMG_to_NUMPY(directory,root):
    path = os.path.join(directory,root)
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(50,50))
    arr = np.array(img,'uint8')
    return arr



x_train = []
y_train = []

x_test = []
y_test = []

TRAIN_PATH = r'training_set'
TEST_PATH = r'test_set' 
FILE_PATH = r'C:\Users\user\Documents\Python Scripts\Machine Learning\Dogs Cats CNN\dataset'

########## Training data
for directories ,_,roots in os.walk(os.path.join(FILE_PATH,TRAIN_PATH)):
    for root in roots:
        y_train.append(FIND_TYPE(root))
        x_train.append(IMG_to_NUMPY(directories,root)) 

########### test data
for directories ,_,roots in os.walk(os.path.join(FILE_PATH,TEST_PATH)):
    for root in roots:
        y_test.append(FIND_TYPE(root))
        x_test.append(IMG_to_NUMPY(directories,root)) 


x_train = np.array(x_train)
x_train = x_train.reshape(-1,50,50,1)

x_test = np.array(x_test)
x_test = x_test.reshape(-1,50,50,1)

y_train = np.array(y_train)
y_test = np.array(y_test)


import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
import os 
import pickle

CATEGORIES = {
    'cat':0,
    'dog':1
}

def FIND_TYPE(val):
    if 'dog' in root:
        return (CATEGORIES['dog'])
    elif 'cat' in root:
        return (CATEGORIES['cat'])

    
def IMG_to_NUMPY(directory,root):
    path = os.path.join(directory,root)
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(50,50))
    arr = np.array(img,'uint8')
    return arr



x_train = []
y_train = []

x_test = []
y_test = []

TRAIN_PATH = r'training_set'
TEST_PATH = r'test_set' 
FILE_PATH = r'C:\Users\user\Documents\Python Scripts\Machine Learning\Dogs Cats CNN\dataset'

########## Training data
for directories ,_,roots in os.walk(os.path.join(FILE_PATH,TRAIN_PATH)):
    for root in roots:
        y_train.append(FIND_TYPE(root))
        x_train.append(IMG_to_NUMPY(directories,root)) 

########### test data
for directories ,_,roots in os.walk(os.path.join(FILE_PATH,TEST_PATH)):
    for root in roots:
        y_test.append(FIND_TYPE(root))
        x_test.append(IMG_to_NUMPY(directories,root)) 


x_train = np.array(x_train)
x_train = x_train.reshape(-1,50,50,1)

x_test = np.array(x_test)
x_test = x_test.reshape(-1,50,50,1)

y_train = np.array(y_train)
y_test = np.array(y_test)


with open('x_train.pickle','wb') as f:
    pickle.dump(x_train,f)

with open('y_train.pickle','wb') as f:
    pickle.dump(y_train,f)

with open('x_test.pickle','wb') as f:
    pickle.dump(x_test,f)

with open('y_test.pickle','wb') as f:
    pickle.dump(y_test,f)
