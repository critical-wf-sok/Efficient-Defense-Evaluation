# SoK: A Critical Evaluation of Efficient Website Fingerprinting Defenses

:warning: :warning: :warning: Experimental - **PLEASE BE CAREFUL**. Intended for Reasearch purposes ONLY. :warning: :warning: :warning:

This repository contains resources of the **SoK: A Critical Evaluation of Efficient Website Fingerprinting Defenses** paper accepted in ***IEEE Symposium on Security and Privacy (Oakland) 2023***. [Read the Paper](https://oaklandsok.github.io/papers/mathews2023.pdf)

## Defenses

In this project, we reevaluated nine website fingerprinting defenses. We did not re-implement any defense; all evaluations were performed using the codes made available by the defense's respective authors. 

Links to the author's defense codes are provided below:

+ [BiMorphing](https://github.com/shibz-islam/BiMorphing/blob/master/countermeasures/BiDiMorphing.py)
+ [FRONT](https://github.com/websitefingerprinting/WebsiteFingerprinting/tree/master/defenses/front)
+ [Deep Fingerprinting Defender](https://github.com/ahmedkas/DFD-FingerPrinting)
+ [Spring & Interspace](https://github.com/pylls/padding-machines-for-tor/tree/master/machines/phase3)
+ [Blind Adversarial Network Pertubations](https://github.com/SPIN-UMass/BLANKET/blob/main/trainer/deepcorr.py)
+ [DynaFlow](https://github.com/davidboxboro/DynaFlow)
+ [Traffic Sliver](https://github.com/TrafficSliver/splitting_simulator)
+ [HyWF / Multihoming](https://github.com/sebhenri/HyWF)

Notes: (1) Most defense simulators are written in Python and will require some adjustments to load and apply on most publicly available datsets. (2) Spring & Interspace simulator requires the Tor project code and compatible datasets such as [the GoodEnough dataset](https://github.com/pylls/padding-machines-for-tor/tree/master/dataset).

## BigEnough (BE) Datasets

Our evaluations were run using a dataset we collected in 2022. Three variants of the dataset was collected for the three safety configuration settings---Standard, Safer, Safest---in the Tor Browser Bundle. Each dataset contains a monitored and unmonitored set of samples. The monitored set of samples contains 19,000 samples collected across 95 websites. Each website is represented using ten subpages from that website. The unmonitored set contains 19,000 samples of index pages for non-monitored websites.

+ This dataset is available to download as Python pickle files [here](https://drive.google.com/drive/folders/15lUs8nimQVTXKG-_WGPLHpcbFhYnIPtH?usp=sharing). The data is stored as processed traffic metadata sequences (timestamp multiplied by direction), and accompanying scripts are provided to aid in preparing the data objects for other projects.

+ Compressed cell trace files (for applying circuit padding simulator defenses) have also been made available [here](https://drive.google.com/drive/folders/1UP6CVqboshx6TYUBIR4y6vAEDQPaPCoT?usp=sharing).

### Collection

All datasets were collected using the data collection scripts developed by Tobias Pylls as part of his padding machines for Tor project. The scripts log Tor cells as they are sent and recieved by the Tor process. The collection of Tor cell logs are allow for defense [simulations using the Tor's circuit padding framework](https://github.com/pylls/circpad-sim). The resources needed to run this tool is available [here](https://github.com/pylls/padding-machines-for-tor/tree/master/collect-traces).

#### TrafficSliver

To accompany our evaluations, we additionally collected an on-network TrafficSliver dataset. To collect this data, we ran a private Tor relay using the [trafficsliver-net](https://github.com/TrafficSliver/trafficsliver-net) Tor process. 

+ The processed dataset is available [here](https://drive.google.com/file/d/1_pIJrMWk35eh7BYlpPGShq1XzY1kI5z8/view?usp=share_link).

+ Our fork of the trafficsliver-net application is available [here](https://github.com/notem/tor-trafficsliver-circpadsim)

## Training & Evaluation

In our evaluation of these defenses, we used the Deep Fingerprinting convolutional neural network model with the Tik-Tok timing x direction input representation for all defenses (with an exception to DynaFlow where we used a different, bespoke traffic sequence representation). Code to reproduce the Tik-Tok attack can be found [here](https://github.com/msrocean/Tik_Tok).

Additional notes:
+ When evaluating defenses with stochastic elements, we generated multiples of defended samples as a data augmentation approach. 
+ We extend this approach for FRONT, where we adjusted the parameters used to generate the defended training data between epochs. Fewer real samples with many more simulated examples were used early during training to help the model learn the padding distribution in an easier environment. This approach however is _extremely_ computationally expensive as new samples need to be generated every epoch.
+ The modified training code used for DynaFlow and FRONT are available in the _training_ subdirectory.

## Information Leakage

The Information Leakage analysis was done using the Python re-implementation of the [WeFDE project code]() that we wrote a few years prior. This repository can be found [here](https://github.com/notem/reWeFDE).

## DynaFlow Prototype

Our paper included a discussion of the implementability of the DynaFlow defense. In the Appendix, we discussed our experiences and results when applying the DynaFlow defense on the real-world Tor network. The Pluggable Transport implementation of DynaFlow defense prepared for that case-study can be found [here](https://github.com/notem/wfpad-dynaflow/blob/main/obfsproxy/transports/wfpadtools/specific/dynaflow.py).
