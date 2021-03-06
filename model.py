import csv
import cv2
import numpy as np
import numpy.random
import sklearn
import math
import argparse
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(basedir+'/'+name)
                if (center_image is not None):
                    center_image = center_image[:,:,::-1]
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def load_data():
    for line in lines:
        image=cv2.imread(line[0])
        steering = float(line[3])
        images.append(image)
        measurements.append(steering)

def nvidia(input_shape):
    print("Nvidia")
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    model = Sequential()
    # compile and train the model using the generator function
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=input_shape))

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Conv2D(24, (16, 16), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Conv2D(36, (8, 8), strides=(2, 2), padding='valid', kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Conv2D(48, (4, 4), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.001), activation='elu'))

    #model.add(Dropout(0.50))

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(0.001), activation='elu'))

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    #model.add(Dropout(0.50))
    model.add(Dense(50, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    #model.add(Dropout(0.50))
    model.add(Dense(10, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    #model.add(Dropout(0.50))

    # Add a fully connected output layer
    model.add(Dense(1))
    model.summary()
    if summaryonly:
        return
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=10)
    model.save('model.nvidia.h5')

def nvidia2(input_shape):
    print("Nvidia modified")
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    model = Sequential()
    # compile and train the model using the generator function
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=input_shape))

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Conv2D(24, (10, 10), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001), activation='elu'))

    #model.add(Dropout(0.50))

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001), activation='elu'))

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    #model.add(Dropout(0.50))
    model.add(Dense(50, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    #model.add(Dropout(0.50))
    model.add(Dense(10, kernel_regularizer=regularizers.l2(0.001), activation='elu'))
    #model.add(Dropout(0.50))

    # Add a fully connected output layer
    model.add(Dense(1))
    model.summary()
    if summaryonly:
        return
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=10)
    model.save('model.nvidia2.h5')

def kasper(input_shape):
    print("Kasper Sakmann model")
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,3)))
    model.add(Conv2D(32, (8,8) ,padding='same', strides=(4,4), activation='relu'))
    model.add(Conv2D(64, (8,8) ,padding='same',strides=(4,4), activation='relu'))
    model.add(Conv2D(128, (4,4),padding='same',strides=(2,2), activation='relu'))
    model.add(Conv2D(128, (2,2),padding='same',strides=(1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dense(1))
    model.summary()
    if summaryonly:
        return
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=5)
    model.save('model.kasper.h5')

def lenet(input_shape):
    print("Lenet model")
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape))
    model.add(Conv2D(6,(5,5),activation=activation))
    model.add(MaxPooling2D())
    model.add(Conv2D(24,(3,3),activation=activation))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.summary()
    if summaryonly:
        return
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=10)
    model.save('model.lenet.h5')

def final_model(input_shape):
    print("Final model")
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape))
#     model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, (8,8) ,padding='same', strides=(4,4),activation=activation))
#     model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (8,8) ,padding='same',strides=(4,4),activation=activation))
#     model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, (4,4),padding='same',strides=(4,4),activation=activation))
    #model.add(Conv2D(128, (2,2),padding='same',strides=(1,1),activation=activation))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128,activation=activation))
    model.add(Dense(128))
    model.add(Dense(1))
    model.summary()
    plot_model(model, to_file='final_model_plot.png', show_shapes=True, show_layer_names=True)
    if summaryonly:
        return
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=7)
    printTraining(history)
    model.save('model.final.h5')

def printTraining(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('final_model_accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('final_model_loss.png')
    
parser = argparse.ArgumentParser(description='Visulization')
parser.add_argument(
    'input_path',
    type=str,
    help='Path data.'
)
args = parser.parse_args()

activation = 'relu'
batch_size=320
lines =[]
shape = None
summaryonly=False
basedir = args.input_path
with open(basedir+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    skip_first=False
    for line in reader:
        if (skip_first):
            lines.append(line)
            if (shape is None):
                image=cv2.imread(basedir+'/'+line[0])
                print(basedir+'/'+line[0])
                shape = image.shape
        else:
            skip_first=True

# images =[]
# measurements=[]
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
# load_data()

# X_train = np.array(images)
# y_train = np.array(measurements)
# lenet(shape)
# nvidia(shape)
# nvidia2(shape)
# kasper(shape)
final_model(shape)


# NVIDIA
# 164s - loss: 0.0184 - val_loss: 0.0104
# 254/254 [==============================] - 25s - loss: 0.0358 - val_loss: 0.0250
# Epoch 2/10
# 254/254 [==============================] - 23s - loss: 0.0228 - val_loss: 0.0228
# Epoch 3/10
# 254/254 [==============================] - 23s - loss: 0.0207 - val_loss: 0.0214
# Epoch 4/10
# 254/254 [==============================] - 22s - loss: 0.0195 - val_loss: 0.0209
# Epoch 5/10
# 254/254 [==============================] - 22s - loss: 0.0185 - val_loss: 0.0206
# Epoch 6/10
# 254/254 [==============================] - 22s - loss: 0.0177 - val_loss: 0.0206
# Epoch 7/10
# 254/254 [==============================] - 22s - loss: 0.0169 - val_loss: 0.0213
# Epoch 8/10
# 254/254 [==============================] - 22s - loss: 0.0165 - val_loss: 0.0218
# Epoch 9/10
# 254/254 [==============================] - 23s - loss: 0.0159 - val_loss: 0.0223
# Epoch 10/10
# 254/254 [==============================] - 22s - loss: 0.0154 - val_loss: 0.0227

# kaspar
# 38572/38572 [==============================] - 43s - loss: 0.0121 - val_loss: 0.0090
# Epoch 2/3
# 38572/38572 [==============================] - 37s - loss: 0.0102 - val_loss: 0.0084
# Epoch 3/3
# 38572/38572 [==============================] - 37s - loss: 0.0093 - val_loss: 0.0070

# kaspar modified
# 503/503 [==============================] - 42s - loss: 0.3350 - val_loss: 0.1798
# Epoch 2/10
# 503/503 [==============================] - 36s - loss: 0.1218 - val_loss: 0.0841
# Epoch 3/10
# 503/503 [==============================] - 36s - loss: 0.0675 - val_loss: 0.0561
# Epoch 4/10
# 503/503 [==============================] - 37s - loss: 0.0503 - val_loss: 0.0468
# Epoch 5/10
# 503/503 [==============================] - 37s - loss: 0.0440 - val_loss: 0.0429
# Epoch 6/10
# 503/503 [==============================] - 36s - loss: 0.0411 - val_loss: 0.0409
# Epoch 7/10
# 503/503 [==============================] - 36s - loss: 0.0395 - val_loss: 0.0395
# Epoch 8/10
# 503/503 [==============================] - 36s - loss: 0.0386 - val_loss: 0.0386
# Epoch 9/10
# 503/503 [==============================] - 37s - loss: 0.0381 - val_loss: 0.0380
# Epoch 10/10
# 503/503 [==============================] - 36s - loss: 0.0377 - val_loss: 0.0378

# lenet
# Epoch 1/10
# 254/254 [==============================] - 17s - loss: 0.0343 - val_loss: 0.0246
# Epoch 2/10
# 254/254 [==============================] - 16s - loss: 0.0240 - val_loss: 0.0232
# Epoch 3/10
# 254/254 [==============================] - 16s - loss: 0.0215 - val_loss: 0.0218
# Epoch 4/10
# 254/254 [==============================] - 16s - loss: 0.0197 - val_loss: 0.0213
# Epoch 5/10
# 254/254 [==============================] - 17s - loss: 0.0184 - val_loss: 0.0214
# Epoch 6/10
# 254/254 [==============================] - 16s - loss: 0.0172 - val_loss: 0.0214
# Epoch 7/10
# 254/254 [==============================] - 17s - loss: 0.0165 - val_loss: 0.0224
# Epoch 8/10
# 254/254 [==============================] - 17s - loss: 0.0154 - val_loss: 0.0221
# Epoch 9/10
# 254/254 [==============================] - 17s - loss: 0.0148 - val_loss: 0.0235
# Epoch 10/10
# 254/254 [==============================] - 16s - loss: 0.0139 - val_loss: 0.0239

# final model
# 506/506 [==============================] - 54s - loss: 0.0347 - val_loss: 0.0249
# Epoch 2/7
# 506/506 [==============================] - 52s - loss: 0.0258 - val_loss: 0.0247
# Epoch 3/7
# 506/506 [==============================] - 54s - loss: 0.0237 - val_loss: 0.0230
# Epoch 4/7
# 506/506 [==============================] - 52s - loss: 0.0227 - val_loss: 0.0227
# Epoch 5/7
# 506/506 [==============================] - 53s - loss: 0.0219 - val_loss: 0.0219
# Epoch 6/7
# 506/506 [==============================] - 52s - loss: 0.0214 - val_loss: 0.0211
# Epoch 7/7
# 506/506 [==============================] - 51s - loss: 0.0208 - val_loss: 0.0209
exit()
