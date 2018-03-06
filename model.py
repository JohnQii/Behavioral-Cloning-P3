import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.models import Model
from sklearn.utils import shuffle
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, Input, BatchNormalization, Dropout
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# LenetTest()
def LenetTest():
    """The Lenet from class videos. not use it after test"""
    images = []
    measurements = []
    path = './data/IMG/'
    lines = []
    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
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

#not use
def getImagesMeasures(path, line, i, correction):
    """Return the image and measurement which is center or left or right """
    name = path + 'IMG/' + line[0].split('/')[-1]
    orgin = cv2.imread(name)
    image = cv2.cvtColor(orgin, cv2.COLOR_BGR2RGB)
    measurement = float(line[3])
    if i==1:
        return (image, measurement + correction)
    elif i==2:
        return (image, measurement - correction)
    else:
        return(image, measurement)


def getLines(path):
    """Get the lines from driving_log.csv and return the lines."""
    lines = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def getPathAngle(lines, path, correction):
    """
    :param lines: lines which from .csv
    :param path: data path
    :param correction: correction for the leaft and right camera
    :return: the path and angles from the lines
    """
    images_path = []
    measurements = []
    for line in lines:
        images_path.append(path + 'IMG/' + line[0].split('/')[-1])
        images_path.append(path + 'IMG/' + line[1].split('/')[-1])
        images_path.append(path + 'IMG/' + line[2].split('/')[-1])
        angle = float(line[3])
        measurements.append(angle)
        measurements.append(angle + correction)
        measurements.append(angle - correction)
    return (images_path, measurements)

def flipImages(images_paths, angles):
    """
    :param images_paths: the path from driving_log.csv
    :param angles: the angles from driving_log.csv
    :return: double the images_path and angles.
    """
    length = len(images_paths)
    flips = []
    for i in range(length):
        flips.append(0)
    for i in range (length):
        images_paths.append(images_paths[i])
        angles.append(angles[i]*(-1))
        flips.append(1)
    return (images_paths, angles, flips )

def generator(samples, batch_size=32):
    """Generator the data"""
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagepath,angle, isflip in batch_samples:
                orgin = cv2.imread(imagepath)
                image = cv2.cvtColor(orgin, cv2.COLOR_BGR2RGB)
                if isflip:
                    images.append(cv2.flip(image, 1))
                else:
                    images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def NVIDIA_net():
    """
    Create the model from Nvida
    :return: model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def NVIDIATrain():
    """Test the "End to End Learning for Self-Driving Cars" by NVIDIA
     http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
     """

    datapath = './data/'
    lines = getLines(datapath)
    images_paths, angles = getPathAngle(lines, datapath, correction=0.2)
    flips = []
    images_paths, angles, flips = flipImages(images_paths, angles)
    # 80% of the data will be used for training
    datas = list(zip(images_paths, angles, flips))
    X_train, X_valid = train_test_split(datas, test_size = 0.2)

    print( 'Length of X_train: ',len( X_train ) )
    print( 'Length of X_valid: ',len( X_valid ) )
    # compile and train the model using the generator function
    train_generator = generator(X_train, batch_size=128)
    validation_generator = generator(X_valid, batch_size=128)

    # Nvidia Network
    model = NVIDIA_net()

    # Use mean squared error for regression, and an Adams optimizer.
    model.compile(loss='mse', optimizer='adam')

    history_object = model.fit_generator(train_generator, samples_per_epoch=len(X_train),
                        validation_data = validation_generator,
                        nb_val_samples = len(X_valid),
                        nb_epoch = 2)
    model.save('model2.h5')
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

NVIDIATrain()