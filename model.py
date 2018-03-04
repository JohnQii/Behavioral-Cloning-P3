import csv
import cv2
import numpy as np
import matplotlib.image as mpimg

#import the data from the csv file
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
path = './data/IMG/'
for line in lines:
    image_center = cv2.imread(path + line[0].split('/')[-1])
    image_left = cv2.imread(path + line[0].split('/')[-1])
    image_right = cv2.imread(path + line[0].split('/')[-1])
    images.append(image_center)
    images.append(image_left)
    images.append(image_right)
    measurement_center = float(line[3])
    correction = 0.2
    measurement_left = measurement_center + correction
    measurement_right = measurement_center - correction
    measurements.append(measurement_center)
    measurements.append(measurement_left)
    measurements.append(measurement_right)

#Flipping Images And Steering Measurements
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement)
    augmented_measurements.append(measurement*(-1.0))

# X_train = np.array(images)
# y_train = np.array(measurements)
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(X_train.shape)
print(y_train.shape)
#train the data
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3))) #normalizing the data and mean centering the data
# model.add(Flatten())
# model.add(Dense(1))
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3))) #normalizing the data and mean centering the data
model.add(Cropping2D(cropping = ((70,25), (0,0)))) #remove the top 70 pixels and the bottom 25
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split= 0.2, shuffle=True, nb_epoch=2) #default epoch= 10

model.save('model.h5')