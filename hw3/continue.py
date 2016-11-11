import pickle, json, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('path', help = 'path to data')
parser.add_argument('model_path', help = 'path to model')
parser.add_argument('output_path', help = 'path to output model')

args = parser.parse_args()

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

path = args.path
model_path = args.model_path
output_path = args.output_path

f = open(path + '/all_label.p', 'rb')

nb_classes = 10
nb_epoch = 30

all_label = pickle.load(f)
f.close()

data = []
answer = np.zeros((5000, 10), dtype=np.float)
index = 0
for index_array in all_label:
    for picture in index_array:
        data.append(np.array(picture, dtype=np.float).reshape(3,32,32))
    
    for j in range(500 * index, 500 * (index + 1)):
        answer[j][index] = 1.0

    index += 1

data = np.array(data)

model = load_model(model_path)

data = data.astype('float32')
data /= 255

model.fit(data, answer, batch_size=32, nb_epoch=nb_epoch, shuffle=True)

model.save(output_path)

