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
        conv_3d = Conv3D(filters = self.filters, kernel_size=3, strides = 1, padding = 'same')(x)
        conv_3d_shape = conv_3d._keras_shape
        print(conv_3d_shape)
        conv_3d = Reshape((conv_3d_shape[1], conv_3d_shape[2], conv_3d_shape[3]*conv_3d_shape[4]))(conv_3d)

        conv_2d = Conv2D(filters = self.filters, kernel_size=3, strides = 1, padding = 'same')(conv_3d)
        conv_2d_shape = conv_2d._keras_shape
        print(conv_2d_shape)

        conv_2d = Reshape((conv_2d_shape[1],conv_2d_shape[2]*conv_2d_shape[3]))(conv_2d)

        conv_1d = Conv1D(filters = self.filters, kernel_size=3, strides = 1, padding = 'same')(conv_2d)
        conv_1d_shape = conv_1d._keras_shape
        print(conv_1d_shape)

        gap = GlobalAveragePooling1D()(conv_1d)
        fc = Dense(self.filters, use_bias = True)(gap)
        softmax = Activation('softmax')(fc)

        reshape_1d = Reshape((1, self.filters))(softmax)
        deconv_1d = Conv1D(filters = self.filters, kernel_size = 3, strides = 1, padding = 'same')(reshape_1d)
        reshape_2d = Reshape((1,1, self.filters))(deconv_1d)
        deconv_2d = Conv2DTranspose(filters = self.filters, kernel_size=3, strides = 1, padding = 'same')(reshape_2d)
        reshape_3d = Reshape((1,1,1, self.filters))(deconv_2d)
        deconv_3d = Conv3DTranspose(filters = self.filters, kernel_size = 3, strides = 1, padding = 'same')(reshape_3d)
        x = tf.multiply(deconv_3d, x)
        return x

def set_params(args):
    args.batch_size = 64
    args.epochs = 200
    return args


def get_model_compiled(shapeinput, num_class, w_decay=0):
    inputs = Input((shapeinput[0],shapeinput[1],shapeinput[2],1))
    filters = [4,4,4,8]
    x = Conv3D(filters=4,use_bias=False,kernel_size=(3,3,5), padding =       'valid',strides = 1)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    for i in range(4):
      x = Conv3D(filters=filters[i],use_bias=False,  kernel_size=(3,3,5),padding = 'valid',strides = 1)(x)
      a1 = AttentionBlock(filters[i])(x)
      #a1 = LeakyReLU()(a1)
      b1 = AttentionBlock(filters[i])(x)
      #b1 = LeakyReLU()(b1)
      x = Add()([a1,b1])
      x = Dropout(0.4)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)

    x = Dropout(0.85)(x)
    x = Flatten()(x)
    x = Dropout(0.85)(x)
    x = Dense(units=128, use_bias=True)(x)
    x = LeakyReLU()(x)
    x = Dense(units=64, use_bias=True)(x)
    x = LeakyReLU()(x)

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
    parser.add_argument('--spatialsize', default=9,
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
        clf.load_weights("/tmp/best_model.h5")
        clf.compile(loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        print("PARAMETERS", clf.count_params())
        stats[pos, :] = mymetrics.reports(
            np.argmax(clf.predict(x_test), axis=1), y_test)[2]
    print(args.dataset, list(stats[-1]))


if __name__ == '__main__':
    main()
