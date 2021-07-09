#!/usr/bin/python

from __future__ import print_function

import argparse

import h5py
import keras
import numpy as np
from keras import backend as K
from keras import regularizers
from keras.constraints import Constraint
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.models import load_model


def my_crossentropy(y_true, y_pred):
    return K.mean(2 * K.abs(y_true - 0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)


def mymask(y_true):
    return K.minimum(y_true + 1., 1.)


def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)


def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10 * K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(
        K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01 * K.binary_crossentropy(y_pred, y_true)), axis=-1)


def my_accuracy(y_true, y_pred):
    return K.mean(2 * K.abs(y_true - 0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''

    def __init__(self, c=2, **kwargs):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-l", "--learning_rate",
                        required=False, type=float, default=0.001, help="learning rate for adam optimizer")
    parser.add_argument("-e", "--epochs",
                        required=False, type=int, default=120, help="learning rate for adam optimizer")
    parser.add_argument("-b", "--base_model_path",
                        required=False, help="base model in h5 format to finetune")
    parser.add_argument("-t", "--training_data_path",
                        required=True, help="training data in h5 format to train on")
    parser.add_argument("-o", "--output_model_path",
                        required=True, default="weights.hdf5", help="path to store the output trained mode in hdf5 format")

    return parser.parse_args()


def main():
    args = parse_arguments()
    reg = 0.000001
    constraint = WeightClip(0.499)

    if not args.base_model_path:
        print('Build model...')
        main_input = Input(shape=(None, 42), name='main_input')
        tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint,
                    bias_constraint=constraint)(main_input)
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

        denoise_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True,
                          name='denoise_gru', kernel_regularizer=regularizers.l2(reg),
                          recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint,
                          recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)

        denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint,
                               bias_constraint=constraint)(denoise_gru)

        model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

    else:
        print(f'Loading base model for finetuning...{args.base_model_path}')
        model = load_model(args.base_model_path, custom_objects={'WeightClip': WeightClip})

    optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(loss=[mycost, my_crossentropy],
                  metrics=[msse],
                  optimizer=optimizer, loss_weights=[10, 0.5])

    batch_size = 32

    print(f'Loading data...{args.training_data_path}')
    with h5py.File(args.training_data_path, 'r') as hf:
        all_data = hf['data'][:]
    print('done.')

    window_size = 2000

    nb_sequences = len(all_data) // window_size
    print(nb_sequences, ' sequences')
    x_train = all_data[:nb_sequences * window_size, :42]
    x_train = np.reshape(x_train, (nb_sequences, window_size, 42))

    y_train = np.copy(all_data[:nb_sequences * window_size, 42:64])
    y_train = np.reshape(y_train, (nb_sequences, window_size, 22))

    noise_train = np.copy(all_data[:nb_sequences * window_size, 64:86])
    noise_train = np.reshape(noise_train, (nb_sequences, window_size, 22))

    vad_train = np.copy(all_data[:nb_sequences * window_size, 86:87])
    vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

    all_data = 0;

    print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)

    print('Train...')
    model.fit(x_train, [y_train, vad_train],
              batch_size=batch_size,
              epochs=args.epochs,
              validation_split=0.1)
    model.save(args.output_model_path)
    print(f'Saved trained model to ... {args.output_model_path}')


if __name__ == '__main__':
    main()
