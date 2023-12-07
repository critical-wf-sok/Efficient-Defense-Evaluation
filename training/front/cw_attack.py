from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import placeholder as Placeholder
import random

from keras.utils import np_utils
from keras.optimizers import Adamax
import numpy as np
import sys
import os
from timeit import default_timer as timer
from pprint import pprint
import argparse
from sklearn import metrics
from data_utils import load_data, generate_defended
import json
from tqdm import tqdm
from pathos.multiprocessing import ProcessPool as Pool
from itertools import repeat
from random import shuffle

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
    parser.add_argument('-l', '--length',
                        type=int,
                        default=8000,
                        help="Length of input vector to model.")
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-s', '--style', default="tiktok")
    return parser.parse_args()


def task(base_cls, input_size, style, use_random, gen_count=100):
    """
    """
    base, cls = base_cls[0], base_cls[1]
    samples, targets = [], []
    if use_random:
        defended, overhead = generate_defended(base, count=-1, style=style, length=input_size)
        samples.extend(defended)
        targets.append(cls)
    else:
        defended, overhead = generate_defended(base, count=gen_count, style=style, length=input_size)
        samples.extend(defended)
        targets.extend([cls for _ in range(gen_count)])
    return samples, targets, overhead


def make_defended_batches(data, sample_indices, input_size, style, batch_size, use_random=False, gen_count=100):
    """
    """
    class_list = data.keys()

    # generate defended training traffic samples
    bases = []
    for cls in class_list:
        for sample_index in sample_indices:
            bases.append((data[cls][sample_index], cls))
    samples = []
    targets = []
    pkt_overheads = []
    with Pool(8) as pool:
        it = pool.imap(task, bases, repeat(input_size), repeat(style), repeat(use_random), repeat(gen_count))
        for s, t, o in tqdm(it, total=len(bases), desc="Make batches..."):
            samples.extend(s)
            targets.extend(t)
            pkt_overheads.append(o)
    #print(pkt_overheads)
    print(f"overhead_avg: {np.mean([o[0] for o in pkt_overheads])}")
    print(f"overhead_tot: {np.sum([o[1] for o in pkt_overheads]) / np.sum([o[2] for o in pkt_overheads])}")

    # convert to numpy arrays
    samples = np.stack(samples)
    targets = np.array(targets)

    # shuffle samples
    s = np.arange(samples.shape[0]).astype(np.int)
    np.random.shuffle(s)
    samples, targets = samples[s], targets[s]

    # split into batches
    batch_count = samples.shape[0] // batch_size
    samples = samples[:batch_count*batch_size]
    targets = targets[:batch_count*batch_size]
    batches = zip(np.split(samples, batch_count), np.split(targets, batch_count))
    return batches, batch_count


def train_model(data, input_size, filepath, 
                train_samples=100, val_samples=10, batch_size=32, 
                epochs=300, stop_thr=30, style='tiktok'):
    """
    """
    class_list = data.keys()

    # Generate hold-out traffic
    test_batches, tcount = make_defended_batches(data, range(train_samples, train_samples+val_samples), 
                                            input_size, style, batch_size, use_random=True)
    test_batches = list(test_batches)

    # build DF model
    model = ConvNet.build(classes=len(class_list), input_shape=(input_size, 1))

    # Op placeholder inputs
    x = Placeholder(shape=(batch_size, input_size, 1), dtype=tf.float32)
    y_true = Placeholder(shape=(batch_size, len(class_list)), dtype=tf.float32)
    
    # Prediction op
    y_pred = model(x)
    
    # Loss op
    loss = K.categorical_crossentropy(y_true, y_pred, from_logits=False)

    # Accuracy op
    accuracy = keras.metrics.categorical_accuracy(y_true, y_pred)
    
    # Operation for getting 
    # gradients and updating weights
    optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.002, momentum=0.3, nesterov=True)

    updates_op = optimizer.get_updates(
        params=model.trainable_weights, 
        loss=loss)
    
    # The graph is created, now we need to call it
    # this would be similar to tf session.run()
    train = K.function(
        inputs=[x, y_true], 
        outputs=[loss], 
        updates=updates_op)
    test = K.function(
        inputs=[x, y_true], 
        outputs=[loss])
    evaluate = K.function(
            inputs=[x, y_true],
            outputs=[accuracy])

    # start training epochs
    val_losses = []
    for epoch in range(epochs):
        print('Epoch %s:' % epoch)

        # generate defended training traffic samples
        # as epoch number increases, increase the number of samples used for defense generation
        if epoch < 2:
            si = list(range(train_samples))
            shuffle(si)
            batches, batch_count = make_defended_batches(data, si[:3], input_size, style, batch_size, gen_count=100)
        elif epoch < 5:
            si = list(range(train_samples))
            shuffle(si)
            batches, batch_count = make_defended_batches(data, si[:12], input_size, style, batch_size, gen_count=25)
        elif epoch < 10:
            si = list(range(train_samples))
            shuffle(si)
            batches, batch_count = make_defended_batches(data, si[:50], input_size, style, batch_size, gen_count=9)
        else:
            # for all remaining epochs, train on FRONT traffic using random
            si = list(range(train_samples))
            shuffle(si)
            batches, batch_count = make_defended_batches(data, si, input_size, style, batch_size, use_random=True)

        # Fancy progress bar
        with tqdm(total=batch_count) as pbar:
    
            # Storing losses for computing mean
            losses_train = []
            accuracy_train = []
    
            # Batch loop
            for batch_samples, batch_targets in batches:

                # convert labels to categorical labels
                batch_targets = keras.utils.to_categorical(batch_targets, len(class_list))
    
                # Running the train graph
                loss_train = train([batch_samples, batch_targets])
                acc_train = evaluate([batch_samples, batch_targets])
                
                # Compute loss mean
                losses_train.append(loss_train[0])
                loss_train_mean = np.mean(losses_train)
                accuracy_train.append(acc_train)
                accuracy_train_mean = np.mean(accuracy_train)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description('loss: %.3f, acc: %.3f' % (loss_train_mean, accuracy_train_mean))

            loss_train_mean = np.mean(losses_train)
            accuracy_train_mean = np.mean(accuracy_train)
    
            # Do validation performance
            losses_test = []
            accuracy_test = []
            for batch_samples, batch_targets in test_batches:

                # convert labels to categorical labels
                batch_targets = keras.utils.to_categorical(batch_targets, len(class_list))
                
                # Evaluation test graph
                loss_test = test([batch_samples, batch_targets])
                acc_test = evaluate([batch_samples, batch_targets])
                
                # Compute test loss mean
                losses_test.append(loss_test[0])
                accuracy_test.append(acc_test)

        loss_test_mean = np.mean(losses_test)
        accuracy_test_mean = np.mean(accuracy_test)

        # print epoch's final metrics
        print("Tr loss,[acc]: %.3f,[%.3f]" % (loss_train_mean, accuracy_train_mean))
        print("Va loss,[acc]: %.3f,[%.3f]\n" % (loss_test_mean, accuracy_test_mean))

        # Early stopping
        val_losses.append(loss_test_mean)
        if min(val_losses) == loss_test_mean:
            model.save(filepath)
        if len(val_losses) >= stop_thr:
            i = np.argmin(val_losses[len(val_losses)-stop_thr::-1])
            if i == stop_thr-1:
                break

    # Reload best model
    del model
    model = load_model(filepath)

    # return trained model
    return model
        

def main():
    """
    """
    # # # # # # # # 
    # Parse arguments
    # # # # # # # # 
    args = parse_arguments()

    # hardcoded params
    train_samples = 170
    val_samples   = 15
    test_samples  = 15
    total_samples = train_samples + val_samples + test_samples

    # Load the dataset
    print("Loading dataset...")
    data = load_data(args.traces)#, max_instances=total_samples)
    print(f"classes: {len(data.keys())}")
    print(f"samples: {np.sum([len(c) for c in data.values()])}")

    # Build and train model
    model = train_model(data, args.length, args.output, 
                            train_samples=train_samples, 
                            val_samples=val_samples, 
                            batch_size=args.batch_size, 
                            style=args.style)

    # generate defended training traffic samples
    print("Start testing trained model...")
    class_list = data.keys()
    samples, targets = [], []
    for cls in class_list:
        print(cls)
        for i in range(test_samples):
            base = data[cls][train_samples+val_samples+i]
            defended, _ = generate_defended(base, count=-1,
                                         style=args.style, 
                                         length=args.length)
            samples.extend(defended)
            targets.append(cls)

    # convert to numpy arrays
    samples = np.stack(samples)
    targets = np.array(targets)
    targets = keras.utils.to_categorical(targets, len(class_list))

    # calculate test scores
    try:
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])
    except:
        pass
    score = model.evaluate(samples, targets, verbose=1)
    print("\n=> Test score:", score[0])
    print("=> Test accuracy:", score[1])


if __name__ == "__main__":
    # execute only if run as a script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
