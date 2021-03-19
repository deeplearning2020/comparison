import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import gc
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras.models import Model
from keras.utils import to_categorical as keras_to_categorical
import numpy as np
import sys


class AttentionBlock(Layer):
    def __init__(self, filters):

        super(AttentionBlock, self).__init__()
        self.filters = filters
        #self.init = RandomNormal()

    def call(self, x):

        max = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
        avg = MaxPooling2D(pool_size=(3, 3), padding='same')(x)

        avgc = Conv2D(self.filters, kernel_size=1, padding='same')(avg)

        avgc = Conv2D(self.filters, kernel_size=1, padding='same')(avgc)

        avgc = Conv2D(self.filters, kernel_size=1, padding='same')(avgc)

        avgc = GlobalAveragePooling2D()(avgc)
        avgc = Dense(self.filters)(avgc)
        avgc = Activation('softmax')(avgc)

        avgc = Reshape((1, self.filters))(avgc)
        avgc = Conv1D(self.filters, kernel_size=3, padding='same')(avgc)
        avgc = Activation('relu')(avgc)
        avgc = Reshape((1, 1, self.filters))(avgc)
        avgc = Conv2D(self.filters, kernel_size=3, padding='same')(avgc)
        #avgc = BatchNormalization()(avgc)
        avgc = Activation('relu')(avgc)

        maxc = Conv2D(self.filters, kernel_size=3, padding='same')(max)

        maxc = Conv2D(self.filters, kernel_size=5, padding='same')(maxc)

        maxc = Conv2D(self.filters, kernel_size=7, padding='same')(maxc)

        maxc = GlobalAveragePooling2D()(maxc)
        maxc = Dense(self.filters)(maxc)
        maxc = Activation('softmax')(maxc)

        maxc = Reshape((1, self.filters))(maxc)
        maxc = Conv1D(self.filters, kernel_size=3, padding='same')(maxc)
        maxc = Activation('relu')(maxc)
        maxc = Reshape((1, 1, self.filters))(maxc)
        maxc = Conv2D(self.filters, kernel_size=3, padding='same')(maxc)
        #maxc = BatchNormalization()(maxc)
        maxc = Activation('relu')(maxc)

        mat_mul = Add()([maxc, avgc, x])
        psi = Conv2D(1, kernel_size=1, padding='same')(mat_mul)
        #psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)

        x = tf.multiply(x, psi)
        return x


def set_params(args):
    args.batch_size = 64
    args.epochs = 10
    return args


def get_model_compiled(shapeinput, num_class, w_decay=0):
    inputs = Input((shapeinput[0],shapeinput[1],shapeinput[2]))
    x = Conv2D(filters=32, kernel_size=(
        3, 3), padding='same', strides=1)(inputs)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(x)
    x = Conv2D(filters=32,kernel_size=(
        3, 3), padding='same', strides=1)(x)
    x = AttentionBlock(32)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(
        3, 3), padding='same', strides=1)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(units=128,kernel_regularizer=regularizers.l2(w_decay))(x)
    x = Activation('relu')(x)
    x = Dense(units=64,kernel_regularizer=regularizers.l2(w_decay))(x)
    x = Activation('relu')(x)

    output_layer = Dense(units=num_class, activation='softmax')(x)
    clf = Model(inputs=inputs, outputs=output_layer)
    clf.compile(loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return clf


def main():
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["IP", "UP", "SV", "UH",
                                 "DIP", "DUP", "DIPr", "DUPr"],
                        help='dataset (options: IP, UP, SV, UH, DIP, DUP, DIPr, DUPr)')
    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--components', default=None,
                        type=int, help='dimensionality reduction')
    parser.add_argument('--spatialsize', default=11,
                        type=int, help='windows size')
    parser.add_argument('--wdecay', default=0.02, type=float,
                        help='apply penalties on layer parameters')
    parser.add_argument('--preprocess', default="standard",
                        type=str, help='Preprocessing')
    parser.add_argument('--splitmethod', default="sklearn",
                        type=str, help='Method for split datasets')
    parser.add_argument('--random_state', default=42, type=int,
                        help='The seed of the pseudo random number generator to use when shuffling the data')
    parser.add_argument('--tr_percent', default=0.1,
                        type=float, help='samples of train set')
    parser.add_argument('--use_val', action='store_true',
                        help='Use validation set')
    parser.add_argument('--val_percent', default=0.1,
                        type=float, help='samples of val set')
    parser.add_argument(
        '--verbosetrain', action='store_true', help='Verbose train')
    #########################################
    parser.add_argument('--set_parameters', action='store_false',
                        help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Number of training examples in one forward/backward pass.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of full training cycle on the training set')
    #########################################

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    if args.set_parameters:
        args = set_params(args)

    pixels, labels, num_class = \
        mydata.loadData(args.dataset, num_components=args.components,
                        preprocessing=args.preprocess)
    pixels, labels = mydata.createImageCubes(
        pixels, labels, windowSize=args.spatialsize, removeZeroLabels=False)
    stats = np.ones((args.repeat, num_class+3)) * -1000.0  # OA, AA, K, Aclass
    for pos in range(args.repeat):
        rstate = args.random_state+pos if args.random_state != None else None
        if args.dataset in ["UH", "DIP", "DUP", "DIPr", "DUPr"]:
            x_train, x_test, y_train, y_test = \
                mydata.load_split_data_fix(
                    args.dataset, pixels)  # , rand_state=args.random_state+pos)
        else:
            pixels = pixels[labels != 0]
            labels = labels[labels != 0] - 1
            x_train, x_test, y_train, y_test = \
                mydata.split_data(
                    pixels, labels, args.tr_percent, rand_state=rstate)

        if args.use_val:
            x_val, x_test, y_val, y_test = \
                mydata.split_data(
                    x_test, y_test, args.val_percent, rand_state=rstate)

        inputshape = x_train.shape[1:]
        clf = get_model_compiled(inputshape, num_class, w_decay=args.wdecay)
        valdata = (x_val, keras_to_categorical(y_val, num_class)) if args.use_val else (
            x_test, keras_to_categorical(y_test, num_class))
        clf.fit(x_train, keras_to_categorical(y_train, num_class),
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=args.verbosetrain,
                validation_data=valdata,
                callbacks=[ModelCheckpoint("/tmp/best_model.h5", monitor='val_accuracy', verbose=0, save_best_only=True)])
        del clf
        K.clear_session()
        gc.collect()
        clf = load_model("/tmp/best_model.h5",custom_objects={'AttentionBlock': AttentionBlock})
        print("PARAMETERS", clf.count_params())
        stats[pos, :] = mymetrics.reports(
            np.argmax(clf.predict(x_test), axis=1), y_test)[2]
    print(args.dataset, list(stats[-1]))


if __name__ == '__main__':
    main()
