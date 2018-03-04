import csv
import cv2
import numpy as np

#import the data from the csv file
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
for line in lines:
    images_center = cv2.imread(line[0])
    images.append(images_center)
    measurement = float(line[3])
    measurements.append(measurement)
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
#train the data
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3))) #normalizing the data and mean centering the data
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split= 0.2, shuffle=True, nb_epoch=2) #default epoch= 10

model.save('model.h5')