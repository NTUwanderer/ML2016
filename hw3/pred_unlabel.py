import pickle, json, argparse
import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

path = 'data'
model_path = 'label_30.h5'

f = open(path + '/all_unlabel.p', 'rb')
all_unlabel = pickle.load(f)
f.close()

nb_classes = 10
nb_epoch = 30

threshold_value = 0.8

training_data = []
for picture in all_unlabel:
    training_data.append(np.array(picture, dtype=np.float).reshape(3,32,32))

training_data = np.array(training_data)

model = load_model(model_path)

training_data = training_data.astype('float32')
training_data /= 255

results = model.predict(training_data, verbose=1)
