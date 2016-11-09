import pickle, json, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('path', help = 'path to data')
parser.add_argument('model_path', help = 'path to model')
parser.add_argument('-o', '--optimizer', type = str, default = 'adam', choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'], help = 'optimizer for model compile')

args = parser.parse_args()

# from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
# from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

path = args.path
model_path = args.model_path
optimizer = args.optimizer

f = open(path + '/all_label.p', 'rb')

batch_size = 32
nb_classes = 10
nb_epoch = 30

all_label = pickle.load(f)
f.close()

data = []
answer = np.zeros((5000, 10), dtype=np.float)

index = 0
for index_array in all_label:
    for picture in index_array[0:500]:
        data.append(np.array(picture, dtype=np.float).reshape(3,32,32))

    for j in range(500 * index, 500 * (index + 1)):
        answer[j][index] = 1.0

    index += 1

data = np.array(data)

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

data = data.astype('float32')
data /= 255

model.fit(data, answer, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)

model.save(model_path)

