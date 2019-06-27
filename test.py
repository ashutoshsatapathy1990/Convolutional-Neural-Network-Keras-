from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
import input
import numpy as np
import os
import cv2

num_classes = 5

# Input image dimensions
img_rows, img_cols = 48, 48
# Images are RGB.
img_channels = 3

####################################LOADING TEST DATASET####################################

# Load the dataset for validation purposes
(x_train, y_train, x_test, y_test) = input.extract()

# Convert class vectors to binary class matrices.
y_test = keras.utils.to_categorical(y_test, num_classes)

x_test = x_test.astype('float32')
x_test /= 255

#############################################END############################################

#####################################LOADING TRAINED MODEL##################################

# Load model and its corresponding weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("weight.h5")

# Let's train the model using RMSprop
ok = model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

if ok is None:
	print("Loaded model from disk")

###########################################END################################################

#################################TEST SET CLASSIFICATION######################################

score = model.predict_classes(x_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

###########################################END################################################

###############################READING IMAGE FOR CLASSIFICATION###############################

img = cv2.imread("#IMAGE PATH")
img = cv2.resize(img, (img_rows, img_cols))
img_test = img.reshape(1, 48, 48, 3)
img_test = img_test.astype('float32')
img_test /= 255

#############################################END##############################################

#################################SINGLE IMAGE CLASSIFICATION##################################

score = model.predict_classes(img_test, verbose=0)
print(score[0])

###########################################END################################################
