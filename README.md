# README

* 1. [Introduction](#1-introduction)
* 2. [Prerequisites](#2-prerequisites)
* 3. [Datasets](#3-datasets)
* 4. [MoStress](#4-mostress)
  * 4.1 [Preprocessing Step](#41-preprocessing-step)
    * 4.1.1 [Config File](#411-config-file)
    * 4.2.2 [Implemented Steps](#412-implemented-steps)
  * 4.2 [Recurrent Neural Network Architecture](#42-recurrent-neural-network-architecture)
* 5. [Improvements](#5-improvements)
* 6. [Future Work](#6-future-work)

## 1. Introduction

This is the implementation of MoStress: a sequence model for stress classification. For more information about the model itself and all the processes and experiments around it, please, read: [MoStress: a Sequence Model for Stress Classification](https://gdoramos.net/publications/). For more questions and discussions, please contact me or any of the authors of the previous article.

Ther resume of MoStress is shown on the image below:

![MoStress](./MoStressFull.jpg)

Also it is importante to mention that this code it doesn't contain exploratory functions or methods, here is just the execution of MoStress.

Although , contributions to add those exploratory features are really welcome, please, create a pull request to add it!

## 2. Prerequisites

The code is written in python 3.7 and the versions of the usage packeges are listed below:

TODO: add list of versions here, or maybe add [shields](https://shields.io/category/platform-support).

At this point in time, MoStress only support [WESAD dataset](https://dl.acm.org/doi/pdf/10.1145/3242969.3242985?casa_token=JPRVzf9hoRAAAAAA:paazllad7xmErtVz4Z5SvhMGKakLlQJCbooGm93uLZXpTvkcsAyzd5QR8071z3Coc8r6qq5EF6s6), therefore, in order to run the code successfully, it is imporant to obtain the WESAD data in advance.

## 3. Datasets

Right now, there is only the implementation of the physiologic data collected from chest sensor of the WESAD. Thus, to help deal with the possible configuration of different datasets, we use the json ```configFiles/wesadDatasetOptions.json``` to add custom configurations and if you want to add new datasets, we suggest to do the same.

If you need to add a new dataset, just create the class which implement it and add the class call on the ```datasets/DatasetFactory.py``` and also make your class inherit the abstract class ```datasets/Dataset.py``` and don't forget to implement the ```_getData()``` method.

## 4. MoStress

### 4.1 Preprocessing Step

#### 4.1.1 Config File

On the folder configFiles, there is a ```configFiles/wesadDatasetOptions.json```, where we set all the parameters needed on the preprocessing.

#### 4.1.2 Implemented Steps

On ```moStress/preprocessing``` we have the main class ```MoStressPreprocessing.py```, which has the call for all the steps that has to be done.

Each step were implemented as classes on ```moStress/preprocessing/implementedSteps``` folder, and they all extends the abstract class: ```moStress/preprocessing/implementedSteps/Steps.py```, so if you want to implement a new step, please, inherit this class also.

### 4.2 Recurrent Neural Network Architecture

In progress...

## 5 Improvements

1. Implement a resume report after the data preprocessing;
2. Implement the k-fold validation to improve the results;
3. Check if it is possible to lower the number of for loops;
4. Implement unit tests to ensure secure changes.

## 6. Future Work

1. Change the RNN for reservoir computing or spiking neural networks;
2. Use N-BEATS, with or without the preprocessing step.
