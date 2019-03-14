import csv
import cv2
import numpy as np
import numpy.random
import sklearn

basedir = 'data'
# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     while 1: # Loop forever so the generator never terminates
#         shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]

#             images = []
#             angles = []
#             for batch_sample in batch_samples:
#                 name = basedir + '/IMG/'+batch_sample[0].split('/')[-1]
#                 center_image = cv2.imread(name)
#                 center_angle = float(batch_sample[3])
#                 images.append(center_image)
#                 angles.append(center_angle)

#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(angles)
#             yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
# batch_size=32

def preprocess_item(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g_max=np.amax(gray)
    g_min=np.amin(gray)
    g = np.divide(np.subtract(gray, g_min), g_max-g_min)
    return np.dstack((g))

def flip(image, steering, r):
#     r = np.random.randint(0,2)
    if (r==0):
        return np.fliplr(image),-steering
    else:
        return image,steering

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def getImage(line, correction, which, isflip):
    if (which==0):
        src_path=line[2]
        steering = float(line[3]) - correction
    elif (which==1):
        src_path=line[1]
        steering = float(line[3]) + correction
    else:
        src_path=line[0]
        steering = float(line[3])
    filename=src_path.split('/')[-1]
    cur_path=basedir+'/IMG/'+filename
    image=cv2.imread(cur_path)    
    image=cv2.resize(image,(80,160))
    cv2.imwrite('a.png',image)
#     image=preprocess_item(cv2.imread(cur_path))
#     print(image.shape)
    image,steering=flip(image,steering, isflip)
#     image = random_brightness(image)
    return image,steering
    
lines =[]
with open(basedir+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    skip_first=False
    for line in reader:
        if (skip_first):
            lines.append(line)
        else:
            skip_first=True

samples = []
# with open(basedir+'/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)

def generateData(iterations):
    for i in range(0, iterations):
        for line in lines:
            image,steering=getImage(line, 0.05, i, iterations%2)
            images.append(image)
            measurements.append(steering)
            break

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

images =[]
measurements=[]
count=0
generateData(6)

X_train = np.array(images)
y_train = np.array(measurements)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# simple regression
# model = Sequential()
# model.add(Flatten(input_shape=(160,320,3)))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

# LeNet with deeper 2nd 
# model = Sequential()
# model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160, 320, 3)))
# model.add(Lambda(lambda x: x/255.0-0.5))
# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(24,3,3,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

# NVIDIA
# model = Sequential()
# model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160, 320, 3)))
# model.add(Lambda(lambda x: x/255.0-0.5))
# model.add(Convolution2D(3,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(24,5,5,activation='relu'))
# model.add(Convolution2D(36,5,5,activation='relu'))
# model.add(Convolution2D(48,3,3,activation='relu'))
# model.add(Convolution2D(64,3,3,activation='relu'))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(300))
# model.add(Dense(50))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)
# 164s - loss: 0.0184 - val_loss: 0.0104

# kaspar 
model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160, 80, 3)))
model.add(Lambda(lambda x: x/127.5 - 1.0))
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)
# 38572/38572 [==============================] - 43s - loss: 0.0121 - val_loss: 0.0090
# Epoch 2/3
# 38572/38572 [==============================] - 37s - loss: 0.0102 - val_loss: 0.0084
# Epoch 3/3
# 38572/38572 [==============================] - 37s - loss: 0.0093 - val_loss: 0.0070
model.save('model.h5')
exit()


