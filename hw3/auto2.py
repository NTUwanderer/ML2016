import pickle, json, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('path', help = 'path to data')
parser.add_argument('model_path', help = 'path to model')
parser.add_argument('output_model_path', help = 'path to output model')
parser.add_argument('-o', '--optimizer', type = str, default = 'adam', choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'], help = 'optimizer for model compile')
parser.add_argument('-e', '--epoch', type = int, default = 30, help = 'nb_epoch')

args = parser.parse_args()

# from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
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

path = args.path
model_path = args.model_path
output_model_path = args.output_model_path
optimizer = args.optimizer
nb_epoch = args.epoch

f = open(path + '/all_label.p', 'rb')

batch_size = 32
nb_classes = 10

all_label = pickle.load(f)
f.close()

data = []
answers = np.zeros((5000, 10), dtype=np.int)

index = 0
for index_array in all_label:
    for picture in index_array[0:500]:
        data.append(np.array(picture, dtype=np.float))

    for j in range(500 * index, 500 * (index + 1)):
        answers[j][index] = 1

    index += 1

data = np.array(data)

data_tf = np.zeros((5000, 3072), dtype=np.float)


data = data.astype('float32')
data /= 255

f_test = open(path + '/all_unlabel.p', 'rb')

x_test = pickle.load(f_test)
f_test.close()

test_size = 45000
x_test = np.array(x_test, dtype=np.float).reshape(test_size, 3072)

x_test /= 255

batch_size_auto = 100
original_dim = 3072
nb_epoch_auto = 10

encoding_dim = 512  # 512 floats -> compression of factor 24.5, assuming the input is 784 floats
layer_dim = 512
numOfLayers = 5

max_size = 5000

model = Sequential()
model.add(Dense(encoding_dim, activation='relu', input_shape=(3072,)))
for i in range(numOfLayers):
    model.add(Dense(layer_dim, activation='relu'))
model.add(Dense(3072, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit(  data, data,
            batch_size=batch_size_auto,
            nb_epoch=nb_epoch_auto,
            verbose=1)
            # validation_data=(x_test[0:3000], x_test[0:3000]))
score = model.evaluate(x_test[0:3000], x_test[0:3000])
print('score', score)

encoder = K.function([model.layers[0].input], [model.layers[2].output])

data_encoded = encoder([data])[0]

average_data = np.zeros((10, layer_dim), dtype=np.float)

for i in range(len(data_encoded)):
    for j in range(layer_dim):
        average_data[floor(i / 500)][j] += data_encoded[i][j]

for i in range(10):
    for j in range(layer_dim):
        average_data[i][j] /= 500

# x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
x_test_encoded = encoder([x_test])[0]

temp_error = np.zeros((10), dtype=np.float)
count = np.zeros((10), np.int)

results = []

for i in range(len(x_test_encoded)):
    temp_index = 0
    for ch in range(10):
        temp_error[ch] = 0
        for j in range(layer_dim):
            temp_error[ch] += (x_test_encoded[i][j] - average_data[ch][j]) ** 2

        if (temp_error[ch] < temp_error[temp_index]):
            temp_index = ch

    result = [i, temp_index, temp_error[temp_index]]
    results.append(result)
    if (i % 1000 == 0):
        print("result: ", result)

def getValue(result):
    return result[2];

print('sorting...')
results = sorted(results, key = getValue)

if (len(results) < max_size):
    max_size = len(results)

new_data = []
new_answers = np.zeros((max_size, 10), dtype=np.int)

weights = [1.0] * 5000
delta = 0.4 / max_size

for i in range(max_size):
    new_data.append(x_test[results[i][0]])
    new_answers[i][results[i][1]] = 1
    weights.append(0.9 - delta * i)

new_data = np.array(new_data)

data = np.concatenate((data, new_data))
answers = np.concatenate((answers, new_answers))
weights = np.array(weights)

data = data.reshape(len(data), 3, 32, 32)

old_model = load_model(model_path)

old_model.fit(data, answers, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, sample_weight=weights)

old_model.save(output_model_path)
