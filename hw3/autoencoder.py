import pickle, json, argparse
import numpy as np

parser = argparse.ArgumentParser()
# parser.add_argument('path', help = 'path to data')
# parser.add_argument('model_path', help = 'path to model')
# parser.add_argument('output_model_path', help = 'path to output model')
parser.add_argument('-o', '--optimizer', type = str, default = 'adam', choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'], help = 'optimizer for model compile')
parser.add_argument('-e', '--epoch', type = int, default = 30, help = 'nb_epoch')

args = parser.parse_args()

# from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
# from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

from keras.layers import Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives

from math import pow, floor

# path = args.path
# model_path = args.model_path
path = 'data'
model_path = 'model/auto_label_30.h5'
optimizer = args.optimizer
nb_epoch = args.epoch

f = open(path + '/all_label.p', 'rb')

batch_size = 32
nb_classes = 10

all_label = pickle.load(f)
f.close()

data = []
answers = np.zeros((5000, 10), dtype=np.int)
# answers = [[0,0,0,0,0,0,0,0,0,0]] * 5000

index = 0
for index_array in all_label:
    for picture in index_array[0:500]:
        data.append(np.array(picture, dtype=np.float))

    for j in range(500 * index, 500 * (index + 1)):
        answers[j][index] = 1

    index += 1

data = np.array(data)

data_tf = np.zeros((5000, 3072), dtype=np.float)

# for i in range(0, 3):
#     for j in range(0, 32):
#         for k in range(0, 32):
#             for n in range(0, 5000):
#                 data_tf[n][j][k][i] = data[n][i][j][k]

# data_tf = data_tf.reshape(5000, 3072)
# encoding_dim = 256

# input_img = Input(shape=(3, 32, 32))

# encoded = Dense(encoding_dim, activation='relu')(input_img)

# decoded = Dense(784, activation='sigmoid')(encoded)

data = data.astype('float32')
data /= 255

f_test = open(path + '/all_unlabel.p', 'rb')

x_test = pickle.load(f_test)
f_test.close()

test_size = 45000
x_test = np.array(x_test, dtype=np.float).reshape(test_size, 3072)

# x_test_tf = np.zeros((test_size, 32, 32, 3), dtype=np.float)

# for i in range(0, 3):
#     for j in range(0, 32):
#         for k in range(0, 32):
#             for n in range(0, test_size):
#                 x_test_tf[n][j][k][i] = x_test[n][i][j][k]

# x_test_tf = x_test_tf.reshape(test_size, 3072)

batch_size = 100
original_dim = 3072
nb_epoch_auto = 50

encoding_dim = 512  # 512 floats -> compression of factor 24.5, assuming the input is 784 floats
layer_dim = 512
numOfLayers = 5

model = Sequential()
model.add(Dense(encoding_dim, activation='relu', input_shape=(3072,)))
for i in range(numOfLayers):
    model.add(Dense(layer_dim, activation='relu'))
model.add(Dense(3072, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit(  data, data,
            batch_size=batch_size,
            nb_epoch=nb_epoch_auto,
            verbose=1,
            validation_data=(x_test[0:3000], x_test[0:3000]))
score = model.evaluate(x_test[0:3000], x_test[0:3000])
print('score', score)

encoder = K.function([model.layers[0].input], [model.layers[2].output])

data_encoded = encoder([data])[0]

# decoded = Dense(original_dim, activation='sigmoid')(encoded)

average_data = np.zeros((10, 256), dtype=np.float)

for i in range(len(data_encoded)):
    for j in range(256):
        average_data[floor(i / 500)][j] += data_encoded[i][j]

for i in range(10):
    for j in range(256):
        average_data[i][j] /= 500

# x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
x_test_encoded = encoder([x_test])[0]

error = np.zeros((len(x_test_encoded)), dtype=np.float)
average_error = np.zeros((10), dtype=np.float)

temp_error = np.zeros((10), dtype=np.float)
count = np.zeros((10), np.int)
for i in range(len(x_test_encoded)):
    temp_index = 0
    for ch in range(10):
        temp_error[ch] = 0
        for j in range(256):
            temp_error[ch] += pow((x_test_encoded[i][j] - average_data[ch][j]), 2)

        if (temp_error[ch] < temp_error[temp_index]):
            temp_index = ch

    average_error[temp_index] += temp_error[temp_index]
    count[temp_index] += 1
    print("error: ", temp_error[temp_index], "index: ", temp_index)

for ch in range(10):
    average_error[ch] /= count[ch]

print(average_error)

