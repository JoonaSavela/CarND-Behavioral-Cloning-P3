import csv
import cv2
import numpy as np
import keras
import sklearn

lines = []
with open('./data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)
    for line in reader:
        if any(field.strip() for field in line):
            lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

data_multiplier = 6
correction_factor = 0.2
resize_factor = 0.5

def generator(samples, batch_size = 8 * data_multiplier):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, int(batch_size / data_multiplier)):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i].replace('\\', '/')
                    name = './data/IMG/'+source_path.split('/')[-1]
                    image = cv2.imread(name)
                    image = cv2.resize(image, (int(320 * resize_factor), int(160 * resize_factor)), interpolation = cv2.INTER_AREA) 
                    correction = (1 / 2) * correction_factor * i * (-3 * i + 5)
                    angle = float(batch_sample[3]) + correction
                    images.append(image)
                    angles.append(angle)
                    flipped_image = np.fliplr(image)
                    flipped_angle = -angle
                    images.append(flipped_image)
                    angles.append(flipped_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

from keras.models import Sequential
from keras.backend import tf
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, ZeroPadding2D

model = Sequential()
model.add(Cropping2D(cropping=((int(65*resize_factor),int(25*resize_factor)), (0,0)), input_shape=(int(160*resize_factor), int(320*resize_factor), 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 4, 4, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(36, 4, 4, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 4, 4, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 2, 2, activation='relu'))
model.add(Convolution2D(64, 1, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * data_multiplier, validation_data=validation_generator, nb_val_samples=len(validation_samples) * data_multiplier, nb_epoch=5)

model.save('model.h5')
print('Model saved')
