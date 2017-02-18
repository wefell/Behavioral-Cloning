import csv
import cv2
import numpy as np
import sklearn
import random
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(lines, batch_size=32):
    num_lines = len(lines)
    while 1:
        shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]
            
            images = []
            measurements = []
            for line in batch_lines:
                for i in range(3):
                    source_path = line[i]
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    local_path = './data/IMG/' + filename
                    image = cv2.imread(local_path)
##                    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    change_pct = random.uniform(0.4, 1.2)
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    hsv[:,:,2] = hsv[:,:,2] * change_pct
                    image_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    images.append(image_brightness)
                correction = 0.2
                measurement = float(line[3])
                measurements.append(measurement)
                measurements.append(measurement+correction)
                measurements.append(measurement-correction)

            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                flipped_image = cv2.flip(image, 1)
                flipped_measurement = float(measurement) * -1.0
                augmented_images.append(flipped_image)
                augmented_measurements.append(flipped_measurement)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)
            

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

total_samples = (len(train_samples)*3)*2

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(30,30))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch=13824,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=15)

model.save('model.h5')

from keras import backend as K
K.clear_session()
