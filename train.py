from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import input

###################################INITIALIZATION########################################

batch_size = 150
num_classes = 26
epochs = 84
data_augmentation = True

# Input image dimensions
img_rows, img_cols = 48, 48
# Images are RGB.
img_channels = 3

# Load the dataset for training and validation purposes
(x_train, y_train, x_test, y_test) = input.extract()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#######################################END#################################################

####################################MODEL CREATION##########################################

model = Sequential()
model.add(Conv2D(48, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

########################################END#####################################################

####################################MODEL TRAINING##############################################

if not data_augmentation:
	print('Not using data augmentation.')
	model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
	print('Using real-time data augmentation.')
    	# This will do preprocessing and realtime data augmentation:
	datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False)  # randomly flip images

	# Compute quantities required for feature-wise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(x_train)
	model.fit_generator(datagen.flow(x_train, y_train,
					batch_size=batch_size),
                        		steps_per_epoch=x_train.shape[0] // batch_size,
	                        	epochs=epochs,
					validation_data=(x_test, y_test))

##############################################END#################################################

#####################################PUBLISHING MODEL#############################################

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)

model.save_weights('weight.h5')

############################################END###################################################
