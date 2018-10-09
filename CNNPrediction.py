import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model
 
 
train_data_dir = 'D:\SHOP PRACTICE PROJECT\data_'
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
img_width, img_height = 150, 150
# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)
 
 
 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_height,img_width, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(64))
 
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
 
 
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_samples = 514
 
epochs = 50
model.fit_generator(
        train_generator,
        steps_per_epoch=1, epochs=epochs)
 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
#img = load_img('F:/Project/Hobe/test')  # this is a PIL image
img = load_img('C:/Users/meesa/Desktop/2.jpg', target_size=(150,150))
 
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict_classes(x)
print
prob = model.predict_proba(x)
print(max(prob))
 
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
print(preds)
