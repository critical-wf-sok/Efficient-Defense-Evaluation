from keras.utils import np_utils
import numpy as np
import os
from data_utils import load_data, make_split
from cw_attack import train_model


def predict(trained_model, X_te_mon, X_te_unmon):
    """
    """
    print("Total testing data: ", len(X_te_mon) + len(X_te_unmon))
    res_mon = trained_model.predict(X_te_mon, verbose=2)
    res_unmon = trained_model.predict(X_te_unmon, verbose=2)
    print("Results count: ", res_unmon.shape[0] + res_mon.shape[0])
    return res_mon, res_unmon


def evaluate(th, y_mon, y_unmon, res_mon, res_unmon):
    """
    """
    print("Testing with threshold = ", th)
    TP = 1
    FP = 1
    TN = 1
    FN = 1

    # ==============================================================
    # Test with Monitored testing instances
    # evaluation
    for i in range(len(res_mon)):
        sm_vector = res_mon[i]
        predicted_class = np.argmax(sm_vector)
        max_prob = max(sm_vector)

        if predicted_class in y_mon: # predicted as Monitored
            if max_prob >= th: # predicted as Monitored and actual site is Monitored
                TP = TP + 1
            else: # predicted as Unmonitored and actual site is Monitored
                FN = FN + 1
        elif predicted_class in y_unmon: # predicted as Unmonitored and actual site is Monitored
            FN = FN + 1

    # ==============================================================
    # Test with Unmonitored testing instances
    # evaluation
    for i in range(len(res_unmon)):
        sm_vector = res_unmon[i]
        predicted_class = np.argmax(sm_vector)
        max_prob = max(sm_vector)

        if predicted_class in y_mon: # predicted as Monitored
            if max_prob >= th: # predicted as Monitored and actual site is Unmonitored
                FP = FP + 1
            else: # predicted as Unmonitored and actual site is Unmonitored
                TN = TN + 1
        elif predicted_class in y_unmon: # predicted as Unmonitored and actual site is Unmonitored
            TN = TN + 1

    print("TP : ", TP)
    print("FP : ", FP)
    print("TN : ", TN)
    print("FN : ", FN)
    print("Total  : ", TP + FP + TN + FN)
    TPR = float(TP) / (TP + FN)
    print("TPR : ", TPR)
    FPR = float(FP) / (FP + TN)
    print("FPR : ",  FPR)
    Precision = float(TP) / (TP + FP)
    print("Precision : ", Precision)
    Recall = float(TP) / (TP + FN)
    print("Recall : ", Recall)
    print("\n")

    return "%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f" % (th, TP, FP, TN, FN, TPR, FPR, Precision, Recall)


def do_ow(model, X_te_mon, y_te_mon, X_te_unmon, y_te_unmon, log_file=None):
    """
    """
    res_mon, res_unmon = predict(model, X_te_mon, X_te_unmon)
    threshold = 1.0 - 1 / np.logspace(0.05, 2, num=15, endpoint=True)

    trials = []
    for th in threshold:
        trials.append(evaluate(th, y_te_mon, y_te_unmon, res_mon, res_unmon))

    if log_file:
        with open(log_file, 'w') as fi:
            fi.write("%s,%s,%s,%s,%s,%s  ,%s  ,  %s, %s\n" % ('Threshold', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR', 'Precision', 'Recall'))
            fi.write('\n'.join(trials))


def main(args):
    """
    """
    print("Loading dataset...")
    X, y = load_data(args.cw_traces, max_length=args.length, max_instances=200)
    y = np.zeros(y.shape)
    classes = len(np.unique(y))
    X_ow, _ = load_data(args.ow_traces, max_length=args.length, fname_pattern=r"(\d+)", max_instances=19000, open_world=True)
    y_ow = np.ones((X_ow.shape[0],))*classes
    print(X.shape, X_ow.shape, y.shape, y_ow.shape)
    X, y = np.concatenate([X, X_ow]), np.concatenate([y, y_ow])
    unmon_class = classes
    classes += 1

    # get split
    X_tr, y_tr, X_te, y_te, X_va, y_va = make_split(X, y, 0.8, 0.1)

    # consider them as float
    X_tr = X_tr.astype('float32')
    X_va = X_va.astype('float32')
    y_tr = y_tr.astype('float32')
    y_va = y_va.astype('float32')
    
    print(X_tr.shape[0], 'training samples')
    print(X_va.shape[0], 'validation samples')
    
    # convert class vectors to binary class matrices
    y_tr = np_utils.to_categorical(y_tr, classes)
    y_va = np_utils.to_categorical(y_va, classes)

    # train OW model
    model = train_model(X_tr, y_tr, X_va, y_va, classes, args.length, args.model_path)

    # prepare OW testing data
    unmon_mask = np.equal(y_te, np.ones((y_te.shape[0],))*unmon_class)
    #y_te = np_utils.to_categorical(y_te, classes)

    X_te_unmon = X_te[unmon_mask]
    y_te_unmon = y_te[unmon_mask]
    X_te_mon = X_te[~unmon_mask]
    y_te_mon = y_te[~unmon_mask]

    do_ow(model, X_te_mon, y_te_mon, X_te_unmon, y_te_unmon, args.log_file)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--length', type=int)
    parser.add_argument('--cw_traces', type=str)
    parser.add_argument('--ow_traces', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--log_file', type=str, default=None)
    args = parser.parse_args()
    main(args)

