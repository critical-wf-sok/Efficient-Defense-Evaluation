import numpy as np 
import argparse
import logging
import sys
import pandas as pd
import os
from os.path import join
from os import makedirs
import constants as ct
from time import strftime
import matplotlib.pyplot as plt
import multiprocessing as mp
import configparser
import time
import datetime
from pprint import pprint
logger = logging.getLogger('ranpad2')
def init_directories():
    # Create a results dir if it doesn't exist yet
    if not os.path.exists(ct.RESULTS_DIR):
        makedirs(ct.RESULTS_DIR)

    # Define output directory
    timestamp = strftime('%m%d_%H%M')
    output_dir = join(ct.RESULTS_DIR, 'ranpad2_'+timestamp)
    makedirs(output_dir)

    return output_dir
def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    # Set logging format
    ch.setFormatter(logging.Formatter(ct.LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)

def parse_arguments():

    conf_parser = configparser.RawConfigParser()
    conf_parser.read(ct.CONFIG_FILE)


    parser = argparse.ArgumentParser(description='It simulates adaptive padding on a set of web traffic traces.')

    parser.add_argument('p',
                        metavar='<traces path>',
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('-format',
                        metavar='<suffix of a file>',
                        default = '',
                        help='suffix of a file.')
    parser.add_argument('-c', '--config',
                        dest="section",
                        metavar='<config name>',
                        help="Adaptive padding configuration.",
                        choices= conf_parser.sections(),
                        default="default")

    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')

    args = parser.parse_args()
    config = dict(conf_parser._sections[args.section])
    config_logger(args)
    return args,config

def load_trace(fdir):
    with open(fdir,'r') as f:
        tmp = f.readlines()
    t = pd.Series(tmp).str.slice(0,-1).str.split('\t',expand = True).astype('float')
    return np.array(t)

def dump(trace, fname):
    global output_dir
    with open(join(output_dir,fname), 'w') as fo:
        for packet in trace:
            fo.write("{:.4f}".format(packet[0]) +'\t' + "{}".format(int(packet[1]))\
                + ct.NL)

def simulate_one(trace, min_wnd=1, max_wnd=14):
    np.random.seed(datetime.datetime.now().microsecond)
    client_wnd = np.random.uniform(min_wnd, max_wnd)
    server_wnd = np.random.uniform(min_wnd, max_wnd)
    return RP(trace, client_wnd, server_wnd)


def simulate_all(trace, num=14, min_wnd=1, max_wnd=14):
    wnd_candidates = np.linspace(min_wnd, max_wnd, num=num)
    i = 0
    traces = []
    for wnd_s in wnd_candidates:
        for wnd_c in wnd_candidates:
            np.random.seed(datetime.datetime.now().microsecond)
            trace = RP(trace, wnd_c, wnd_s)
            traces.append(trace)
    return traces


def simulate(fdir):
    if not os.path.exists(fdir):
        return
    # logger.debug("Simulating trace {}".format(fdir))
    np.random.seed(datetime.datetime.now().microsecond)
    trace = load_trace(fdir)
    trace = RP(trace)
    fname = fdir.split('/')[-1]
    dump(trace, fname)

def RP(trace, client_wnd=None, server_wnd=None):
    # format: [[time, pkt],[...]]
    # trace, cpkt_num, spkt_num, cwnd, swnd
    global client_dummy_pkt_num 
    global server_dummy_pkt_num 
    global min_wnd 
    global max_wnd 
    global start_padding_time
    global client_min_dummy_pkt_num
    global server_min_dummy_pkt_num
    
    #if client_wnd is None:
    #    client_wnd = np.random.uniform(min_wnd, max_wnd)
    #if server_wnd is None:
    #    server_wnd = np.random.uniform(min_wnd, max_wnd)
    #if client_min_dummy_pkt_num != client_dummy_pkt_num:
    client_dummy_pkt = np.random.randint(1,2500)#client_min_dummy_pkt_num,client_dummy_pkt_num)
    #client_dummy_pkt = np.random.randint(1,833)#client_min_dummy_pkt_num,client_dummy_pkt_num)
    #else:
    #    client_dummy_pkt = client_dummy_pkt_num
    #if server_min_dummy_pkt_num != server_dummy_pkt_num:
    server_dummy_pkt = np.random.randint(1,2500)#server_min_dummy_pkt_num,server_dummy_pkt_num)
    #server_dummy_pkt = np.random.randint(1,833)#server_min_dummy_pkt_num,server_dummy_pkt_num)
    #else:
    #    server_dummy_pkt = server_dummy_pkt_num
    logger.debug("client_wnd:",client_wnd)
    logger.debug("server_wnd:",server_wnd)
    logger.debug("client pkt:", client_dummy_pkt)
    logger.debug("server pkt:", server_dummy_pkt)


    #first_incoming_pkt_time = trace[np.where(trace[:,1] < 0)][0][0]
    first_incoming_pkt_time = trace[0][0] #[np.where(trace[:,1] < 0)][0][0]
    last_pkt_time = trace[-1][0]    
    
    client_timetable = getTimestamps(client_wnd, client_dummy_pkt)
    #client_timetable = client_timetable[np.nonzero(start_padding_time+client_timetable[:,0] < last_pkt_time)]
    client_timetable = client_timetable[np.nonzero(client_timetable[:,0] < last_pkt_time)]
    #for i,t in enumerate(client_timetable):
    #    if t > last_pkt_time:
    #        break
    #client_timetable = client_timetable[:i]

    server_timetable = getTimestamps(server_wnd, server_dummy_pkt)
    server_timetable[:,0] += first_incoming_pkt_time
    #server_timetable = server_timetable[np.nonzero(start_padding_time+server_timetable[:,0] < last_pkt_time)]
    server_timetable = server_timetable[np.nonzero(server_timetable[:,0] < last_pkt_time)]
    #for i,t in enumerate(server_timetable):
    #    if t > last_pkt_time:
    #        break
    #server_timetable = server_timetable[:i]

    
    # print("client_timetable")
    # print(client_timetable[:10])
    client_pkts = np.concatenate((client_timetable, 512*np.ones((len(client_timetable),1))),axis = 1)
    server_pkts = np.concatenate((server_timetable, -512*np.ones((len(server_timetable),1))),axis = 1)


    noisy_trace = np.concatenate( (trace, client_pkts, server_pkts), axis = 0)
    noisy_trace = noisy_trace[ noisy_trace[:, 0].argsort(kind = 'mergesort')]
    return noisy_trace

def getTimestamps(wnd, num):
    # timestamps = sorted(np.random.exponential(wnd/2.0, num))   
    # print(wnd, num)
    # timestamps = sorted(abs(np.random.normal(0, wnd, num)))
    timestamps = sorted(np.random.rayleigh(wnd,num))
    # print(timestamps[:5])
    # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
    return np.reshape(timestamps, (len(timestamps),1))


def parallel(flist, n_jobs = 20):
    pool = mp.Pool(n_jobs)
    pool.map(simulate, flist)


conf_parser = configparser.RawConfigParser()
conf_parser.read(ct.CONFIG_FILE)
config = dict(conf_parser._sections['default'])

global client_dummy_pkt_num 
global server_dummy_pkt_num 
global client_min_dummy_pkt_num
global server_min_dummy_pkt_num
global max_wnd
global min_wnd 
global start_padding_time
client_min_dummy_pkt_num = int(config.get('client_min_dummy_pkt_num',1))
server_min_dummy_pkt_num = int(config.get('server_min_dummy_pkt_num',1))
client_dummy_pkt_num = int(config.get('client_dummy_pkt_num',2500))
server_dummy_pkt_num = int(config.get('server_dummy_pkt_num',2500))
start_padding_time = int(config.get('start_padding_time', 0))
max_wnd = float(config.get('max_wnd',14))
min_wnd = float(config.get('min_wnd',1))
MON_SITE_NUM = int(config.get('mon_site_num', 100))
MON_INST_NUM = int(config.get('mon_inst_num', 600))
UNMON_SITE_NUM = int(config.get('unmon_site_num', 100))


if __name__ == '__main__':
    #global client_dummy_pkt_num 
    #global server_dummy_pkt_num 
    #global client_min_dummy_pkt_num
    #global server_min_dummy_pkt_num
    #global max_wnd
    #global min_wnd 
    #global start_padding_time
    args, config = parse_arguments()
    logger.info(args)

    client_min_dummy_pkt_num = int(config.get('client_min_dummy_pkt_num',1))
    server_min_dummy_pkt_num = int(config.get('server_min_dummy_pkt_num',1))
    client_dummy_pkt_num = int(config.get('client_dummy_pkt_num',300))
    server_dummy_pkt_num = int(config.get('server_dummy_pkt_num',300))
    start_padding_time = int(config.get('start_padding_time', 0))
    max_wnd = float(config.get('max_wnd',10))
    min_wnd = float(config.get('min_wnd',10))
    MON_SITE_NUM = int(config.get('mon_site_num', 10))
    MON_INST_NUM = int(config.get('mon_inst_num', 10))
    UNMON_SITE_NUM = int(config.get('unmon_site_num', 100))
    print("client_min_dummy_pkt_num:{}".format(client_min_dummy_pkt_num))
    print("server_min_dummy_pkt_num:{}".format(server_min_dummy_pkt_num))
    print("client_dummy_pkt_num: {}\nserver_dummy_pkt_num: {}".format(client_dummy_pkt_num,server_dummy_pkt_num))
    print("max_wnd: {}\nmin_wnd: {}".format(max_wnd,min_wnd))
    print("start_padding_time:", start_padding_time)
    flist  = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            flist.append(join(args.p, str(i)+'-'+str(j)+args.format))
    for i in range(UNMON_SITE_NUM):
        flist.append(join(args.p, str(i)+args.format))

    # Init run directories
    output_dir = init_directories()
    logger.info("Traces are dumped to {}".format(output_dir))
    start = time.time()
    # for i,f in enumerate(flist):
    #     logger.debug('Simulating {}'.format(f))
    #     if i %2000 == 0:
    #         print(r"Done for inst ",i,flush = True)
    #     simulate(f)

    parallel(flist)
    logger.info("Time: {}".format(time.time()-start))
