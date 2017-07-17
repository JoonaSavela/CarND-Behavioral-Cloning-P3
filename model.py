import csv
import cv2
import numpy as np
import keras

lines = []
with open('./data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)
    for line in reader:
        lines.append(line)

print(lines[0][0])
print(lines[0][3])

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    try:
        measurement = float(line[3])
        images.append(image)
        measurements.append(measurement)
    except:
        print(source_path)
        print(current_path)
        print(line[3])
        pass

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)
print(y_train.shape)
print(X_train[0].shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=X_train[0].shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
print('Model saved')


