import pickle, json, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('path', help = 'path to data')
parser.add_argument('model_path', help = 'path to model')
parser.add_argument('output_model_path', help = 'path to output model')
parser.add_argument('-t', '--threshold', type = float, default = 0.8, help = 'set threshold')
parser.add_argument('-w', '--weight', type = int, default = 1, help = 'weight or not')

args = parser.parse_args()

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

path = args.path
model_path = args.model_path
output_model_path = args.output_model_path
threshold_value = args.threshold
weight = args.weight

f = open(path + '/all_label.p', 'rb')
all_label = pickle.load(f)
f.close()

f = open(path + '/all_unlabel.p', 'rb')
all_unlabel = pickle.load(f)
f.close()

nb_classes = 10
nb_epoch = 30

data = []

weights = [1.0] * 5000

answers = np.zeros((5000, 10), dtype=np.float)
index = 0
for index_array in all_label:
    for picture in index_array:
        data.append(np.array(picture, dtype=np.float).reshape(3,32,32))
    
    for j in range(500 * index, 500 * (index + 1)):
        answers[j][index] = 1.0

    index += 1

data = np.array(data)

training_data = []
for picture in all_unlabel:
    training_data.append(np.array(picture, dtype=np.float).reshape(3,32,32))

training_data = np.array(training_data)

model = load_model(model_path)

data = data.astype('float32')
data /= 255

training_data = training_data.astype('float32')
training_data /= 255

# results = model.predict(training_data, verbose=1)
results = np.load(path + '/predictOfUnlabel_label30.npy')

sample_matrix = []
for i in range(0, 10):
    temp = [0] * 10
    temp[i] = 1
    sample_matrix.append(temp)


predict_data = []
predicts = []

result_index = 0
for result in results:
    i = 0
    value = result[i]
    for j in range(1, 10):
        if (result[j] > value):
            i = j
            value = result[i]

    if (value >= threshold_value):
        predict_data.append(training_data[result_index])
        predicts.append(sample_matrix[i])
        # predicts.append(result)
        weights.append(value)

    result_index += 1

print('Num of extracted unlabel data: ', len(predict_data))

print('top 20 predicts: ', predicts[0:20])
print('top 20 weights: ', weights[5000:5020])

predict_data = np.array(predict_data)
predicts = np.array(predicts)

data = np.concatenate((data, predict_data), axis = 0)
answers = np.concatenate((answers, predicts), axis = 0)

if weight:
    weights = np.array(weights)
else:
    weights = None

model.fit(data, answers, batch_size=32, nb_epoch=nb_epoch, shuffle=True, sample_weight=weights)

model.save(output_model_path)

