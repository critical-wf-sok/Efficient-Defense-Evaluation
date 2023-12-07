from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, Input, Concatenate
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import random
from tensorflow.keras import utils as np_utils
from tensorflow.keras.optimizers import Adamax
import numpy as np
import sys
import os
from timeit import default_timer as timer
from pprint import pprint
import argparse
from sklearn import metrics
from data_utils import load_data, make_split
import json

random.seed(0)


# define the ConvNet
class ConvNet:
    @staticmethod
    def build(classes,
              input_shape,
              activation_function=("elu", "relu", "relu", "relu", "relu", "relu"),
              dropout=(0.1, 0.1, 0.1, 0.1, 0.5, 0.7),
              filter_num=(32, 64, 128, 256),
              kernel_size=8,
              conv_stride_size=1,
              pool_stride_size=4,
              pool_size=8,
              fc_layer_size=(512, 512)):

        # confirm that parameter vectors are acceptable lengths
        assert len(filter_num) + len(fc_layer_size) <= len(activation_function)
        assert len(filter_num) + len(fc_layer_size) <= len(dropout)

        # Sequential Keras model template
        model = Sequential()

        # add convolutional layer blocks
        for block_no in range(0, len(filter_num)):
            if block_no == 0:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 input_shape=input_shape,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))
            else:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act1'.format(block_no)))

            model.add(Conv1D(filters=filter_num[block_no],
                             kernel_size=kernel_size,
                             strides=conv_stride_size,
                             padding='same',
                             name='block{}_conv2'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act2'.format(block_no)))

            model.add(MaxPooling1D(pool_size=pool_size,
                                   strides=pool_stride_size,
                                   padding='same',
                                   name='block{}_pool'.format(block_no)))

            model.add(Dropout(dropout[block_no], name='block{}_dropout'.format(block_no)))

        # flatten output before fc layers
        model.add(Flatten(name='flatten'))

        # add fully-connected layers
        for layer_no in range(0, len(fc_layer_size)):
            model.add(Dense(fc_layer_size[layer_no],
                            kernel_initializer=glorot_uniform(seed=0),
                            name='fc{}'.format(layer_no)))

            model.add(BatchNormalization())
            model.add(Activation(activation_function[len(filter_num)+layer_no],
                                 name='fc{}_act'.format(layer_no)))

            model.add(Dropout(dropout[len(filter_num)+layer_no],
                              name='fc{}_drop'.format(layer_no)))

        # add final classification layer
        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc_final'))
        model.add(Activation('softmax', name="softmax"))

        # compile model with Adamax optimizer
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss="categorical_crossentropy",
                      #optimizer=optimizer,
                      optimizer='sgd',
                      metrics=["accuracy"])
        return model


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train and test the DeepFingerprinting model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--traces',
                        type=str,
                        default='traces',
                        metavar='<trace_data>',
                        help='Path to the directory where the training data is stored.')
    parser.add_argument('-o', '--output',
                        type=str,
                        default='DF.h5',
                        metavar='<output>',
                        help='Location to store the file.')
    parser.add_argument('-q', '--quiet',
                        action="store_true",
                        help="Lower verbosity of output.")
    return parser.parse_args()


def train_model(X_tr, y_tr, X_va, y_va, classes, filepath):
    """
    """
    #try:
    #    model = load_model(filepath)
    #except:
    input_shapes = [X.shape[1:] for X in X_tr]
    print("Compiling model...")
    model = ConvNet.build(classes=classes, input_shapes=input_shapes, 
              #activation_function=("elu", "relu", "relu", "relu"),
              #dropout=(0.1, 0.1, 0.5, 0.7),
              #filter_num=(32, 64, 128),
              #pool_size=1,
              fc_layer_size=(512, 512),
    )
    print(model.summary())

    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='auto', restore_best_weights=True)
    callbacks_list = [checkpoint, early_stopping]

    history = model.fit(X_tr, y_tr,
                        epochs=3000,
                        verbose=1,
                        validation_data=(X_va, y_va),
                        callbacks=callbacks_list)

    # Save & reload model
    model.save(filepath)
    del model
    model = load_model(filepath)

    return model

def main():
    """
    """
    # # # # # # # # 
    # Parse arguments
    # # # # # # # # 
    args = parse_arguments()


    # # # # # # # # 
    # Load the dataset
    # # # # # # # # 
    print("Loading dataset...")
    X, y, = load_data(args.traces, max_instances=200, separator='\t')

    # split data into 8:1:1 ratio cuts
    X_tr, y_tr, X_te, y_te, X_va, y_va = make_split(X, y, 0.8, 0.1)

    # consider them as float and normalize
    X_tr = X_tr.astype('float32')
    X_va = X_va.astype('float32')
    X_te = X_te.astype('float32')
    y_tr = y_tr.astype('float32')
    y_va = y_va.astype('float32')
    y_te = y_te.astype('float32')

    NB_CLASSES = len(set(list(y_tr)))
    print('Classes:', NB_CLASSES)

    # convert class vectors to binary class matrices
    y_tr = np_utils.to_categorical(y_tr, num_classes=NB_CLASSES)
    y_va = np_utils.to_categorical(y_va, num_classes=NB_CLASSES)
    y_te = np_utils.to_categorical(y_te, num_classes=NB_CLASSES)

    #print("Shape:", X_tr.shape)
    print(X_tr_c.shape[0], 'train samples')
    print(X_va_c.shape[0], 'validation samples')
    print(X_te_c.shape[0], 'test samples')

    # # # # # # # # 
    # Build and train model
    # # # # # # # # 
    model = train_model(X_tr, y_tr, X_va, y_va, NB_CLASSES, args.output)

    # # # # # # # # 
    # Test the model
    # # # # # # # # 
    score = model.evaluate(X_te, y_te,
                           verbose=1)
    score_train = model.evaluate(X_tr, y_tr,
                                 verbose=1)

    # # # # # # # # 
    # Print results
    # # # # # # # # 
    print("\n=> Train score:", score_train[0])
    print("=> Train accuracy:", score_train[1])

    print("\n=> Test score:", score[0])
    print("=> Test accuracy:", score[1])


if __name__ == "__main__":
    # execute only if run as a script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
