import os
import numpy as np
import re
from tqdm import tqdm
import math
import scipy
from scipy.stats import kstest
import random
from random import shuffle
import pickle
from main import simulate_all, simulate_one, getTimestamps



def generate_defended(base, length, count=225, style="direction"):
    """
    """
    base = np.array([[base[0][i], base[1][i]*base[2][i]] for i in range(len(base[0]))])
    if count <= 1:
        traces = [simulate_one(base)]
        overhead = len(traces[0]) / len(base)
        overhead = (overhead, len(traces[0]), len(base))
    else:
        num = math.ceil(math.sqrt(count))
        # make simulated traces
        traces = simulate_all(base, num)
        overhead = np.mean([len(trace)/len(base) for trace in traces]) 
        overhead = (overhead, len(traces[0]), len(base))

    # convert defended traffic into chose sequence representations
    defended = []
    for trace in traces:
        seq = [[], [], []]
        for packet in trace:
            timestamp = float(packet[0])
            size = abs(int(packet[1]))
            direction = int(packet[1]) // size
            seq[0].append(timestamp)
            seq[1].append(size)
            seq[2].append(direction)
        rep = np.array(process_trace(seq, style))
        # pad trace length with zeros
        if length is not None and style is not "raw":
            if len(rep) < length:
                rep = np.hstack((rep, np.zeros((length - len(rep),))))
            rep = rep[:length].reshape((length, 1))
        defended.append(rep)
    return defended, overhead


def process_trace(trace, style, **kwargs):

    def process_by_direction(trace):
        rep = [0, 0]
        burst_count = 0
        for i in range(len(trace[2])):
            direction = trace[2][i]
            if direction > 0:
                if len(rep)//2 == burst_count:
                    rep.extend([0, 0])
                rep[-2] += 1
            else:
                if len(rep)//2 != burst_count:
                    burst_count += 1
                rep[-1] += 1
        return rep

    def process_by_iat(trace, threshold=0.003):
        burst_seq = [0, 0]
        direction = trace[2][0]
        if direction > 0:
            burst_seq[0] += 1
        else:
            burst_seq[1] += 1
        for i in range(1, len(trace[0])):
            iat = trace[0][i] - trace[0][i-1]
            if iat > threshold:
                burst_seq.extend([0, 0])
            if trace[2][i] > 0:
                burst_seq[len(burst_seq)-2] += 1
            else:
                burst_seq[len(burst_seq)-1] += 1
        return burst_seq

    def process_by_timeslice(trace, interval=0.05):
        burst_seq = [0, 0]
        s = 0
        for i in range(len(trace[0])):
            timestamp = trace[0][i]
            if timestamp // interval > s:
                s = timestamp // interval
                burst_seq.extend([0, 0])
            direction = trace[2][i]
            if direction > 0:
                burst_seq[len(burst_seq)-2] += 1
            else:
                burst_seq[len(burst_seq)-1] -= 1
        return burst_seq

    def process_tiktok(trace):
        return np.array(trace[0])*np.array(trace[2])


    # process into burst sequences using IATs
    if style=='iat':
        rep = process_by_iat(trace, **kwargs)

    # process into bursts by timeslice
    if style=='time':
        rep = process_by_timeslice(trace, **kwargs)

    # process into bursts by direction
    if style=='direction':
        rep = process_by_direction(trace)

    if style == 'tiktok':
        rep = process_tiktok(trace)

    if style=='raw':
        rep = trace

    return rep


def load_trace(fi, separator="\t", filter_by_size=True):
    """
    loads data to be used for predictions
    """
    sequence = [[], [], []]
    for line in fi:
        pieces = line.strip("\n").split(separator)
        if int(pieces[1]) == 0:
            break
        timestamp = float(pieces[0])
        length = abs(int(pieces[1]))
        direction = int(pieces[1]) // length
        if filter_by_size:
            if length > 512:
                sequence[0].append(timestamp)
                sequence[1].append(length)
                sequence[2].append(direction)
        else:
            sequence[0].append(timestamp)
            sequence[1].append(length)
            sequence[2].append(direction)
    return sequence


#def load_data(directory, separator='\t', fname_pattern=r"(\d+)[-](\d+)", 
#        max_classes=99999, max_instances=500, open_world=False, ow_label=-1):
#    """
#    """
#    # load pickle file if it exist
#    data = dict()
#    class_counter = dict()
#    for root, dirs, files in os.walk(directory):
#        # filter for trace files
#        if not open_world:
#            files = [fname for fname in files if re.match(fname_pattern, fname)]
#        for fname in tqdm(files, total=len(files)):
#
#            # get trace class
#            if not open_world:
#                cls = int(re.match(fname_pattern, fname).group(1))
#            else:
#                cls = ow_label
#
#            # skip if maximum number of instances reached
#            if class_counter.get(cls, 0) >= max_instances:
#                continue
#
#            # skip if maximum number of classes reached
#            if int(cls) >= max_classes:
#                continue
#
#            with open(os.path.join(root, fname), "r") as fi:
#
#                # load the trace file
#                trace = load_trace(fi, separator, filter_by_size=False)
#
#                # add to dictionary dictionary
#                if cls not in data.keys():
#                    data[cls] = []
#                data[cls].append(trace)
#                class_counter[int(cls)] = class_counter.get(cls, 0) + 1
#
#    # shuffle samples
#    for cls in data.keys():
#        np.random.shuffle(data[cls])
#
#    # return trace dataset as dictionary
#    return data

def load_data(mon_pkl, unmon_pkl=None):
    data = {i:[] for i in range(95)}
    with open(mon_pkl, 'rb') as fi:
        mon_data = pickle.load(fi)
    #for i in range(2000):
    for i in range(950):
        label = i // 10
        samples = mon_data[i]
        for sample in samples:
            trace = [[],[],[]]
            for cell in sample:
                ts = abs(cell)
                le = 512
                dr = 1 if cell > 0 else -1
                trace[0].append(ts)
                trace[1].append(le)
                trace[2].append(dr)
            data[label].append(trace)
        shuffle(data[label])
    if unmon_pkl:
        data2 = {}
        with open(unmon_pkl, 'rb') as fi:
            unmon_data = pickle.load(fi)
        unmon_cls = max(data.keys())+1
        data2[unmon_cls] = []
        for sample in unmon_data:
            if len(data2[unmon_cls]) >= 19000:
                break
            trace = [[],[],[]]
            for cell in sample:
                ts = abs(cell)
                le = 512
                dr = 1 if cell > 0 else -1
                trace[0].append(ts)
                trace[1].append(le)
                trace[2].append(dr)
            data2[unmon_cls].append(trace)
        shuffle(data2[unmon_cls])
        return data, data2
    return data
