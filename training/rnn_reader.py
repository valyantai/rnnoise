#!/usr/bin/python
# requires pip install tensorflow==1.5
# requires pip install keras==2.3.0
# for latest TF reset_after=False for GRU layers

import sys

import keras
from keras import backend as K
import numpy as np
from keras import regularizers
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model
from keras.constraints import Constraint

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

reg = 0.000001

def read_vector(model_file):
    model_vector = [int(i) for i in model_file.readline().split()]
    return model_vector

def read_layer_shape(model_file, is_gru=False):
    header = model_file.readline()
    weights_0_shape_0, weights_0_shape_1, activation = header.split()
    weights_0_shape_0 = int(weights_0_shape_0)
    weights_0_shape_1 = int(weights_0_shape_1)
    if is_gru:
        weights_0_shape_1 = int(weights_0_shape_1)*3
    return (weights_0_shape_0, weights_0_shape_1)

def float_weights(weights, shape):
    weights = np.array(weights)
    float_weights = np.true_divide(weights, 256)
    reshaped_weights = float_weights.reshape(shape)
    return reshaped_weights

def float_bias(bias):
    bias = np.array(bias)
    float_bias = np.true_divide(bias, 256)
    return float_bias

# Layer shape
# | Num |    Layer    |   Kernel  | Recurrent Kernel |  Bias  |
# |  1  |    dense    |  (42, 24) | ---------------- | (24,)  |
# |  2  |    vad gru  |  (24, 72) |     (24, 72)     | (72,)  |
# |  4  |  noise gru  | (90, 144) |     (48, 144)    | (144,) |
# |  6  | denoise gru |(114, 288) |     (96, 288)    | (288,) |
# |  7  | denoise out |  (96, 22) | ---------------- | (22,)  |
# |  8  |  vad output |  (24, 1)  | ---------------- |  (1,)  |

dense_layer_map = {
    1 : ["input_dense",  (42, 24)],
    7 : ["denoise_output", (96, 22)],
    8 : ["vad_output", (24, 1)]
}

gru_layer_map = {
    2 : ["vad_gru", (24, 72), (24, 72)],
    4 : ["noise_gru", (90, 144), (48, 144)],
    6 : ["denoise_gru", (114, 288), (96, 288)]
}

def rnnoise_model_from_file(file_path):
    constraint = WeightClip(0.499)
    main_input = Input(shape=(None, 42), name='main_input')

    tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(
        main_input)
    vad_gru = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru',
                  kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg),
                  kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
    vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint,
                       bias_constraint=constraint)(vad_gru)

    noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
    noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru',
                    kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg),
                    kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(
        noise_input)

    denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])
    denoise_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru',
                      kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg),
                      kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(
        denoise_input)
    denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint,
                           bias_constraint=constraint)(denoise_gru)

    model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

    with open(file_path, 'r') as model_file:
        print(model_file.readline() )# remove header

        for index in range(9):
            if index in dense_layer_map:
                # Load weights, bias for dense layers
                layer_name = dense_layer_map[index][0]

                weights_shape =  read_layer_shape(model_file) #dense_layer_map[index][1]
                weights = read_vector(model_file)
                bias = read_vector(model_file)
                model.layers[index].set_weights([float_weights(weights, weights_shape), float_bias(bias)])
            elif index in gru_layer_map:
                # Load weights, recurrent weights, bias for gru layers
                layer_name = gru_layer_map[index][0]

                weights_shape = read_layer_shape(model_file, is_gru=True) #gru_layer_map[index][1]
                weights = read_vector(model_file)

                recurrent_weights_shape = gru_layer_map[index][2]
                recurrent_weights = read_vector(model_file)

                bias = read_vector(model_file)

                model.layers[index].set_weights([float_weights(weights, weights_shape), \
                                                 float_weights(recurrent_weights, recurrent_weights_shape), \
                                                 float_bias(bias)])
            else:
                print("{} LAYER DOES NOT NEED TO LOAD WEIGHTS...".format(index))
    # output h5 file, ready to be used in rnn_train.py
    model.save(sys.argv[2])

if __name__ == '__main__':
    rnnoise_model_from_file(sys.argv[1])