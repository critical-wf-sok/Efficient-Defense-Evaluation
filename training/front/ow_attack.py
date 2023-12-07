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
    parser.add_argument('-m', '--mon',
                        type=str,
                        default='traces',
                        metavar='<trace_data>',
                        help='Path to the directory where the training data is stored.')
    parser.add_argument('-u', '--unmon',
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
        defended, _ = generate_defended(base, count=-1, style=style, length=input_size)
        samples.extend(defended)
        targets.append(cls)
    else:
        defended, _ = generate_defended(base, count=gen_count, style=style, length=input_size)
        samples.extend(defended)
        targets.extend([cls for _ in range(len(defended))])
    return samples, targets


def make_defended_batches(data, sample_indices, input_size, style, batch_size, use_random=False, gen_count=100, do_batchify=True):
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
    with Pool(8) as pool:
        it = pool.imap(task, bases, repeat(input_size), repeat(style), repeat(use_random), repeat(gen_count))
        for s, t in tqdm(it, total=len(bases), desc="Make batches..."):
            samples.extend(s)
            targets.extend(t)

    # convert to numpy arrays
    samples = np.stack(samples)
    targets = np.array(targets)

    # shuffle samples
    s = np.arange(samples.shape[0]).astype(np.int)
    np.random.shuffle(s)
    samples, targets = samples[s], targets[s]

    if do_batchify:

        # split into batches
        batch_count = samples.shape[0] // batch_size
        samples = samples[:batch_count*batch_size]
        targets = targets[:batch_count*batch_size]
        batches = zip(np.split(samples, batch_count), np.split(targets, batch_count))
        return batches, batch_count

    else:
        return (samples, targets), None


def train_model(data_mon, data_unmon, input_size, filepath, 
                train_samples=(100, 1000), val_samples=(10, 100), batch_size=32, 
                epochs=300, stop_thr=30, style='tiktok'):
    """
    """
    class_list = data_mon.keys()

    # Generate hold-out traffic
    mon_test_batches, _ = make_defended_batches(data_mon, range(train_samples[0], train_samples[0]+val_samples[0]), 
                                            input_size, style, batch_size, use_random=True)
    unmon_test_batches, _ = make_defended_batches(data_unmon, range(train_samples[1], train_samples[1]+val_samples[1]), 
                                            input_size, style, batch_size, use_random=True)
    test_batches = list(mon_test_batches) + list(unmon_test_batches)

    # build DF model
    model = ConvNet.build(classes=len(class_list)+1, input_shape=(input_size, 1))

    # Op placeholder inputs
    x = Placeholder(shape=(batch_size, input_size, 1), dtype=tf.float32)
    y_true = Placeholder(shape=(batch_size, len(class_list)+1), dtype=tf.float32)
    
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
            s_count = 3
            gen_count = 100

            mon_si = list(range(train_samples[0]))
            shuffle(mon_si)
            unmon_si = list(range(train_samples[1]))
            shuffle(unmon_si)

            mon_batches, c1 = make_defended_batches(data_mon, mon_si[:s_count], input_size, style, batch_size, gen_count=gen_count)
            unmon_batches, c2 = make_defended_batches(data_unmon, mon_si[:s_count], input_size, style, batch_size, gen_count=gen_count)
            batches = list(mon_batches) + list(unmon_batches)
            batch_count = c1 + c2
        elif epoch < 5:
            s_count = 12
            gen_count = 64

            mon_si = list(range(train_samples[0]))
            shuffle(mon_si)
            unmon_si = list(range(train_samples[1]))
            shuffle(unmon_si)

            mon_batches, c1 = make_defended_batches(data_mon, mon_si[:s_count], input_size, style, batch_size, gen_count=gen_count)
            unmon_batches, c2 = make_defended_batches(data_unmon, mon_si[:s_count], input_size, style, batch_size, gen_count=gen_count)
            batches = list(mon_batches) + list(unmon_batches)
            batch_count = c1 + c2
        elif epoch < 10:
            s_count = 25
            gen_count = 25

            mon_si = list(range(train_samples[0]))
            shuffle(mon_si)
            unmon_si = list(range(train_samples[1]))
            shuffle(unmon_si)

            mon_batches, c1 = make_defended_batches(data_mon, mon_si[:s_count], input_size, style, batch_size, gen_count=gen_count)
            unmon_batches, c2 = make_defended_batches(data_unmon, mon_si[:s_count], input_size, style, batch_size, gen_count=gen_count)
            batches = list(mon_batches) + list(unmon_batches)
            batch_count = c1 + c2
        else:
            # for all remaining epochs, train on FRONT traffic using both random and uniform parameters
            mon_si = list(range(train_samples[0]))
            shuffle(mon_si)
            unmon_si = list(range(train_samples[1]))
            shuffle(unmon_si)

            batches = []
            batch_count = 0

            mon_batches, c1 = make_defended_batches(data_mon, mon_si, input_size, style, batch_size, use_random=True)
            unmon_batches, c2 = make_defended_batches(data_unmon, unmon_si, input_size, style, batch_size, use_random=True)
            batches = list(mon_batches) + list(unmon_batches)
            batch_count = c1 + c2
        
        #ran_batches, ran_batch_count = make_defended_batches(data_mon, mon_si, input_size, style, batch_size, 
        #                                                     use_random=True, do_batchify=False)
        #mon_flat_x = ran_batches[0]
        #mon_flat_y = ran_batches[1]
        #ran_batches, ran_batch_count = make_defended_batches(data_unmon, unmon_si, input_size, style, batch_size, 
        #                                                     use_random=True, do_batchify=False)
        #unmon_flat_x = ran_batches[0]
        #unmon_flat_y = ran_batches[1]
        #samples = np.concatenate((mon_flat_x, unmon_flat_x), axis=0)
        #targets = np.concatenate((mon_flat_y, unmon_flat_y), axis=0)
        # shuffle samples
        #s = np.arange(samples.shape[0]).astype(np.int)
        #np.random.shuffle(s)
        #samples, targets = samples[s], targets[s]
        #batch_count = samples.shape[0] // batch_size
        #samples = samples[:batch_count*batch_size]
        #targets = targets[:batch_count*batch_size]
        #batches = zip(np.split(samples, batch_count), np.split(targets, batch_count))

        # Fancy progress bar
        with tqdm(total=batch_count) as pbar:
    
            # Storing losses for computing mean
            losses_train = []
            accuracy_train = []
    
            # Batch loop
            for batch_samples, batch_targets in batches:

                # convert labels to categorical labels
                batch_targets = keras.utils.to_categorical(batch_targets, len(class_list)+1)
    
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
                batch_targets = keras.utils.to_categorical(batch_targets, len(class_list)+1)
                
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


def ow_evaluation(model, mon_samples, unmon_samples, unmon_label):
    """
    """
    upper_bound = 1.0
    #thresholds = upper_bound - upper_bound / np.logspace(0.05, 2, num=15, endpoint=True)
    thresholds = np.linspace(0.05, 0.99, num=25, endpoint=True)

    fmt_str = '{}:\t{}\t{}\t{}\t{}\t{}\t{}'
    print(fmt_str.format('TH  ', 'TP   ', 'TN   ', 'FP   ', 'FN   ', 'Pre. ', 'Rec. '))
    fmt_str = '{:.2f}:\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}'

    # evaluate model performance at different thresholds
    # high threshold will yield higher precision, but reduced recall
    results = []
    for th in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0

        # Test with Monitored testing instances
        for s in range(mon_samples.shape[0]):
            test_example = mon_samples[s]
            predict_prob = model.predict(test_example[np.newaxis, ...])
            best_n = np.argsort(predict_prob[0])[-1:]
            if best_n[0] != unmon_label:
                if predict_prob[0][best_n[0]] >= th:
                    TP = TP + 1
                else:
                    FN = FN + 1
            else:
                FN = FN + 1

        # Test with Unmonitored testing instances
        for s in range(unmon_samples.shape[0]):
            test_example = unmon_samples[s]
            predict_prob = model.predict(test_example[np.newaxis, ...])
            best_n = np.argsort(predict_prob[0])[-1:]
            if best_n[0] != unmon_label:
                if predict_prob[0][best_n[0]] >= th:
                    FP = FP + 1
                else:
                    TN = TN + 1
            else:
                TN = TN + 1

        pre = TP/(TP+FP) if (TP+FP) != 0 else 0
        rec = TP/(TP+FN) if (TP+FN) != 0 else 0
        res = [th, TP, TN, FP, FN, pre, rec]
        print(fmt_str.format(*res))
        results.append(res)
        with open('./front_ow.log', 'a') as fi:
            fi.write(fmt_str.format(*res)+'\n')

    return results
        

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
    unmon_train_samples = 16150
    unmon_val_samples = 1425
    unmon_test_samples = 1425
    unmon_total_samples = unmon_train_samples + unmon_val_samples + unmon_test_samples

    # Load the dataset
    print("Loading dataset...")
    data_mon, data_unmon = load_data(args.mon, args.unmon)
    unmon_label = list(data_unmon.keys())[0]
    print(unmon_label)

    # Build and train model
    model = train_model(data_mon, data_unmon, args.length, args.output, 
                        train_samples=(train_samples, unmon_train_samples), 
                        val_samples=(val_samples, unmon_val_samples), 
                        batch_size=args.batch_size, style=args.style)

    # generate defended training traffic samples
    print("Start testing trained model...")
    class_list = data_mon.keys()
    mon_samples = []
    for cls in class_list:
        for i in range(test_samples):
            base = data_mon[cls][train_samples+val_samples+i]
            defended, _ = generate_defended(base, count=-1,
                                         style=args.style, 
                                         length=args.length)
            mon_samples.extend(defended)
    mon_samples = np.array(mon_samples)
    # generate defended unmon test samples
    unmon_samples = []
    for i in range(unmon_test_samples):
        base = data_unmon[unmon_label][unmon_train_samples+unmon_val_samples+i]
        defended, _ = generate_defended(base, count=-1,
                                     style=args.style, 
                                     length=args.length)
        unmon_samples.extend(defended)
    unmon_samples = np.array(unmon_samples)

    # calculate test scores
    try:
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])
    except:
        pass
    ow_evaluation(model, mon_samples, unmon_samples, unmon_label)


if __name__ == "__main__":
    # execute only if run as a script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
