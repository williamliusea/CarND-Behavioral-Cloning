import csv
import cv2
import numpy as np
import numpy.random
import sklearn

basedir = '/opt/data'
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
# batch_size=32
    
def load_data():
    for line in lines:
        image=cv2.imread(line[0])
        steering = float(line[3])
        images.append(image)
        measurements.append(steering)

# from sklearn.model_selection import train_test_split
# train_samples, validation_samples = train_test_split(samples, test_size=0.2)

lines =[]
with open(basedir+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    skip_first=False
    for line in reader:
        if (skip_first):
            lines.append(line)
        else:
            skip_first=True

images =[]
measurements=[]
load_data()

X_train = np.array(images)
y_train = np.array(measurements)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(80,80,3)))
    model.add(Convolution2D(3,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(24,5,5,activation='relu'))
    model.add(Convolution2D(36,5,5,activation='relu'))
    model.add(Convolution2D(48,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(300))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)
    model.save('model.nvidia.h5')

def kasper():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(80,80,3)))
    model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4),activation='relu'))
    model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4),activation='relu'))
    model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2),activation='relu'))
    model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1),activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
    model.save('model.kasper.h5')
 
def lenet():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(80,80,3)))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(24,3,3,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
    model.save('model.lenet.h5')
   
lenet()
nvidia()
kasper()


# NVIDIA
# 164s - loss: 0.0184 - val_loss: 0.0104

# kaspar 
# 38572/38572 [==============================] - 43s - loss: 0.0121 - val_loss: 0.0090
# Epoch 2/3
# 38572/38572 [==============================] - 37s - loss: 0.0102 - val_loss: 0.0084
# Epoch 3/3
# 38572/38572 [==============================] - 37s - loss: 0.0093 - val_loss: 0.0070
exit()


