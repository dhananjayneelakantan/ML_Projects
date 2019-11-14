
#Common imports

from __future__ import division, print_function, unicode_literals

import os

import numpy as np

import pickle

np.random.seed(42)

import pandas as pd

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")

 

from PIL import Image

import os, sys

from glob import glob                                                          

import cv2

 

 

 

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

 

import matplotlib.image as mpimg

import cv2

from PIL import Image

 

PROJECT_ROOT_DIR = "."

IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "Images")

DOC_PATH = os.path.join(PROJECT_ROOT_DIR, "Data_and_models")

 

TEMP_RESULT_PATH = os.path.join(PROJECT_ROOT_DIR, "Temp_Results")

FINAL_RESULT_PATH = os.path.join(PROJECT_ROOT_DIR, "Final_Results")

DIGIT_RESULT_PATH = os.path.join(PROJECT_ROOT_DIR, "Digits")

 

 

def save_fig(fig_id, tight_layout=True, fig_extension="jpg", resolution=90):

    path = os.path.join(DIGIT_RESULT_PATH, fig_id + "." + fig_extension)

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)

   

def loadImages(path):

    # Put files into lists and return them as one list of size 4

    image_files = sorted([os.path.join(file)          

         for file in os.listdir(path) if(file.endswith('.JPG') or file.endswith('.jpg')) ])

    return image_files

 

 

# Display one image

def display_one(a, title1 = "Original"):

    plt.imshow(a), plt.title(title1)

    plt.xticks([]), plt.yticks([])

    plt.show()

# Display two images

def display(a, b, title1 = "Original", title2 = "Edited"):

    plt.subplot(121), plt.imshow(a), plt.title(title1)

    plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(b), plt.title(title2)

    plt.xticks([]), plt.yticks([])

    plt.show()

 

def detect_left_edge(image,resultpath,filename):

    resultpath = resultpath

    filename = filename

    h,w = image.shape

    max = 0

    edge = 0

    for x in range(0,100):

 

        vertical_slice = image[0:h,  x:x+15 ]

        vertical_slice_pixels_count = vertical_slice.sum()

 

        if( vertical_slice_pixels_count > THRESHOLD_PIXELS_COUNT):

            imageio.imwrite(resultpath + '/' + filename, image[0:h, x:2400])

            return 0

 

        if (vertical_slice_pixels_count > max):

            max = vertical_slice_pixels_count

            edge = x * 2

    return edge

 

def preprocess(image):

    image = image

    max_output_value = 255

    neighorhood_size = 99

    subtract_from_mean = 10

    image_binarized = cv2.adaptiveThreshold(image,

                                            max_output_value,

                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

                                            cv2.THRESH_BINARY,

                                            neighorhood_size,

                                            subtract_from_mean)

 

        #Sharpen image

    kernel = np.array([[0, -1, 0],

                        [-1, 5,-1],

                        [0, -1, 0]])

    image_sharp = cv2.filter2D(image_binarized, -1, kernel)

    #enhance image

    image_enhanced = cv2.equalizeHist(image_sharp)

    #invert image

    imageinvert = cv2.bitwise_not(image_enhanced)

    return imageinvert

 

def resize_digits(w,h):

    for i in range(0,11):

        path = (DIGIT_RESULT_PATH + '/' + str(i) +'/')

        dirs = os.listdir( path )

        for item in dirs:

            if (item.endswith('.JPG') or item.endswith('.jpg') or item.endswith('.png')):

                if os.path.isfile(path+item):   

                    im = Image.open(path+item)

                    f, e = os.path.splitext(path+item)

                    imResize = im.resize((w,h), Image.ANTIALIAS)

                    imResize.save(f + '.jpg', 'JPEG', quality=90)

 

 

 

 

 

 

# Read the input image in greyscale

#image = cv2.imread((IMAGES_PATH + image_list[num]),cv2.IMREAD_GRAYSCALE)

# image = cv2.imread((IMAGES_PATH + image_list[num]))

# image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# image_grey = cv2.cvtColor(image_color,cv2.COLOR_RGB2GRAY)

 

 

from sklearn import datasets, svm, metrics

import scipy.misc

import imageio

 

 

PRINT_SLICES = False

THRESHOLD_PIXELS_COUNT = 60000

MAX_BOUNDING_BOX_WIDTH = 2400

MAX_BOUNDING_BOX_HEIGHT = 650

 

 

image={}

image_list=loadImages(IMAGES_PATH)

for i in range(len(image_list[lt:ut])):

    image[i] = cv2.imread((IMAGES_PATH + '/' + image_list[i]),cv2.IMREAD_GRAYSCALE)

    input_image = image[i]

    image_height, image_width = input_image.shape

    max = 0;

    output_image = input_image

   

    # detect top edge of the image bounding box

    for h in range(0, image_height - MAX_BOUNDING_BOX_HEIGHT):

        temp_image = input_image[h:h + MAX_BOUNDING_BOX_HEIGHT, 0:0 + MAX_BOUNDING_BOX_WIDTH]

        if temp_image.sum() > max:

            max= temp_image.sum()

            output_image = temp_image

    edge = detect_left_edge(output_image,TEMP_RESULT_PATH,image_list[i])

 

 

 

image_preprocessed = {}

temp_image_list=loadImages(TEMP_RESULT_PATH)

 

for i in range(len(temp_image_list)):

 

    image_preprocessed[i] = cv2.imread((TEMP_RESULT_PATH + '/' + temp_image_list[i]),cv2.IMREAD_GRAYSCALE)

 

    image = image_preprocessed[i]

    # Apply adaptive thresholding

 

    imageinvert = preprocess(image)

   

    #set boundary

    PRINT_SLICES = False

    THRESHOLD_PIXELS_COUNT = 60000

    MAX_BOUNDING_BOX_WIDTH = 1200

    MAX_BOUNDING_BOX_HEIGHT = 325

 

    #image input

    input_image = imageinvert

 

    image_height, image_width = input_image.shape

    max = 0;

    output_image = input_image

    # detect top edge of the image bounding box

    for h in range(0, image_height - MAX_BOUNDING_BOX_HEIGHT):

        temp_image = input_image[h:h + MAX_BOUNDING_BOX_HEIGHT, 0:0 + MAX_BOUNDING_BOX_WIDTH]

        if temp_image.sum() > max:

            max= temp_image.sum()

            output_image = temp_image

    edge = detect_left_edge(output_image,FINAL_RESULT_PATH,temp_image_list[i])

 

 

 

PATH = FINAL_RESULT_PATH

image_preprocessed = {}

temp_image_list=loadImages(PATH)

for i in range(len(temp_image_list)):

    image_preprocessed[i] = cv2.imread((PATH + '/' + temp_image_list[i]))

    display_one(image_preprocessed[i])

 

    digit = {}

for i in range(0,10):

    digit_list=loadImages(DIGIT_RESULT_PATH + '/' + str(i) +'/')

    for j in range(len(digit_list)):

        digit[j] = cv2.imread((DIGIT_RESULT_PATH + '/' + str(i) + '/' + digit_list[j]),cv2.IMREAD_GRAYSCALE)

        #digit[j] = cv2.imread((DIGIT_RESULT_PATH + '/' + str(i) + '/' + digit_list[j]))

        #a = preprocess(digit[j])

        #imageinvert = cv2.bitwise_not(a)

 

        display_one(digit[j] )

        print(digit[j].shape)

 

 

 

import numpy as np

import os

import scipy.ndimage

from skimage.feature import hog

from skimage import data, color, exposure

from sklearn.model_selection import  train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.externals import joblib

 

features_list = []

features_label = []

# load labeled training / test data

# loop over the 10 directories where each directory stores the images of a digit

for digit in range(0,10):

    label = digit

    training_directory = (DIGIT_RESULT_PATH + '/' + str(label) + '/')

    for filename in os.listdir(training_directory):

        print(filename)

        if (filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.jpg')):

            training_digit_image = cv2.imread((training_directory + filename),cv2.IMREAD_GRAYSCALE)

            #training_digit_image = color.rgb2gray(training_digit_image)

            display_one(training_digit_image)

            print(training_digit_image.shape)

 

            from skimage.feature import hog

            df= hog(training_digit_image, orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))

           

            features_list.append(df)

            features_label.append(label)

 

 

# store features array into a numpy array

features  = np.array(features_list, 'float64')

# split the labled dataset into training / test sets

X_train, X_test, y_train, y_test = train_test_split(features, features_label)

# train using K-NN

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

# get the model accuracy

model_score = knn.score(X_test, y_test)

 

 

# save trained model

joblib.dump(knn, 'knn_model.pkl')

 

 

#Prediction

knn = joblib.load('knn_model.pkl')

def feature_extraction(image):

    processed_image = preprocess(image)

    #resize to 50X50

    return hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(10, 10), cells_per_block=(5, 5))

def predict(df):

    predict = knn.predict(df.reshape(1,-1))[0]

    predict_proba = knn.predict_proba(df.reshape(1,-1))

    return predict#, predict_proba[predict]

digits = []

# load your image from file

digi=loadImages(DIGIT_RESULT_PATH+'/10')

for i in range(len(digi)):

    digits = cv2.imread((DIGIT_RESULT_PATH + '/10/' +  digi[i]),cv2.IMREAD_GRAYSCALE)

    display_one(digits)

    hogs = (feature_extraction(digits))

    print(predict(hogs))

   

# extract featuress

#hogs = list(map(lambda x: feature_extraction(x), digits))

 

# apply k-NN model created in previous

#predictions = predict(hogs)