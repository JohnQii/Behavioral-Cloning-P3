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

#import the data from the csv file
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#From classroom test, not use it after test.
def LenetTest():
    """The Lenet from class videos"""
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

# LenetTest()


#Generators
def generator(samples, batch_size=128, path = './data/IMG/', correction = 0.05):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # name = path + batch_sample[0].split('/')[-1]
                # center_image = cv2.imread(name)
                # center_angle = float(batch_sample[3])
                # images.append(center_image)
                # angles.append(center_angle)
                for i in range (3):
                    name = path + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    if i == 2:
                        angle -= correction
                    elif i == 1:
                        angle += correction
                    images.append(image)
                    angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def conv2d_BN(input_x, filters, rows, cols, border_mode='same', strides=(1, 1)):
    """ Convolution2D and BatchNormalization
    """
    input_x = Convolution2D(filters, rows, cols, activation='relu', subsample=strides,
                            border_mode=border_mode)(input_x)
    input_x = BatchNormalization()(input_x)
    return input_x

def NVIDIA_net():
    input = Input(shape = (160, 320, 3))
    x = Lambda(lambda x: x/255.0 - 0.5)(input)
    x = Cropping2D(cropping=((70, 25), (0, 0)))(x)
    x = conv2d_BN(x, 24, 5, 5, strides=(2, 2))
    x = conv2d_BN(x, 36, 5, 5, strides=(2, 2))
    x = conv2d_BN(x, 48, 5, 5, strides=(2, 2))
    x = conv2d_BN(x, 64, 3, 3)
    x = conv2d_BN(x, 64, 3, 3)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(100)(x)
    x = Dense(50)(x)
    x = Dense(10)(x)
    x = Dense(1)(x)
    return Model(input, x)

def NVIDIATrain():
    """Test the "End to End Learning for Self-Driving Cars" by NVIDIA
     http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
     """
    # 80% of the data will be used for training.
    X_train, X_valid = train_test_split(lines, test_size = 0.2)
    print( 'Length of X_train: ',len( X_train ) )
    print( 'Length of X_valid: ',len( X_valid ) )
    # compile and train the model using the generator function
    train_generator = generator(X_train)
    validation_generator = generator(X_valid)

    # Nvidia Network
    model = NVIDIA_net()

    # Use mean squared error for regression, and an Adams optimizer.
    model.compile(loss='mse', optimizer='adam')

    history_object = model.fit_generator(train_generator, samples_per_epoch=len(X_train),
                        validation_data = validation_generator,
                        nb_val_samples = len(X_valid),
                        nb_epoch = 3)
    model.save('model.h5')
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