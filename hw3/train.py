import pickle, json, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('path', help = 'path to data')
parser.add_argument('model_path', help = 'path to model')
parser.add_argument('-o', '--optimizer', type = str, default = 'adam', choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'], help = 'optimizer for model compile')
parser.add_argument('-e', '--epoch', type = int, default = 30, help = 'nb_epoch')

args = parser.parse_args()

# from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
# from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

path = args.path
model_path = args.model_path
optimizer = args.optimizer
nb_epoch = args.epoch

f = open(path + '/all_label.p', 'rb')

batch_size = 32
nb_classes = 10

all_label = pickle.load(f)
f.close()

data = []
answers = np.zeros((4000, 10), dtype=np.int)
# answers = [[0,0,0,0,0,0,0,0,0,0]] * 5000

val_data = []
val_answers = np.zeros((1000, 10), dtype=np.int)

index = 0
for index_array in all_label:
    for picture in index_array[0:400]:
        data.append(np.array(picture, dtype=np.float).reshape(3,32,32))

    for picture in index_array[400:500]:
        val_data.append(np.array(picture, dtype=np.float).reshape(3,32,32))


    for j in range(400 * index, 400 * index + 400):
        answers[j][index] = 1

    for j in range(100 * index, 100 * index + 100):
        answers[j][index] = 1

    index += 1

data = np.array(data)
val_data = np.array(val_data)
data = data.astype('float32')
val_data = val_data.astype('float32')
data /= 255
val_data /= 255

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

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



model.fit(data, answers, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, validation_data=(val_data, val_answers))

model.save(model_path)

