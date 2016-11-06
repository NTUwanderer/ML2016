import pickle, json, argparse
import numpy as np

# from keras.models import model_from_json

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

parser = argparse.ArgumentParser()
parser.add_argument('path', help = 'path to data')
parser.add_argument('model_path', help = 'path to trained model')
parser.add_argument('output_path', help = 'path to output csv')

args = parser.parse_args()

path = args.path
model_path = args.model_path
output_path = args.output_path
# path = 'data'
# model_path = 'first_model.h5'

f = open(path + '/test.p', 'rb')

nb_classes = 10
nb_epoch = 200

test_data = pickle.load(f)
f.close()

data = []
index = 0
for picture in test_data['data']:
    data.append(np.array(picture, dtype=np.float).reshape(3,32,32))

data = np.array(data)

data = data.astype('float32')
data /= 255

model = load_model(model_path)

results = model.predict(data, verbose=1)

total_value = 0.0

output = open(output_path, 'w')
output.write('ID,class\n')

confident = open('data/confident.csv', 'w')
confident.write('ID,confident\n')

result_index = 0
for result in results:
    i = 0
    value = result[i]
    for j in range(1, 10):
        if (result[j] > value):
            i = j
            value = result[i]
    output.write(str(result_index) + ',' + str(i) + '\n')
    confident.write(str(result_index) + ',' + str(value) + '\n')
    result_index += 1

    total_value += value

output.close()
confident.close()

print('average value: ', total_value / len(results))

