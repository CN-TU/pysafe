#!/home/fmeghdouri/venv3/bin/python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
import random
from tqdm import tqdm
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree
from sklearn.neighbors import DistanceMetric
from datetime import datetime
import argparse
import pickle
import re
import os
import json

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

# TODO: add plotting code
#        add nn

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

SEED = 2020

def z_scaling(data):
    # correct way to do it is to scale all folds separately
    sc = StandardScaler()
    return sc.fit_transform(data)

def minmax_scaling(data):
    pass

def build_nn(n_layers, n_neurons, activations, loss='binary_crossentropy', optimizer='adam'):
    model = Sequential()
    model.add(Dense(n_neurons[0], input_dim=X_test.shape[-1], activation=activations[0]))
    for i in range(1, n_layers-1):
        model.add(Dense(n_neurons[i], activation=activations[i]))
    model.add(Dense(1, activation=activations[-1]))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def load_data(path):
    print('##### rading data')
    data = pd.read_csv(path).fillna(0)
    y = data['Label'].values
    X = data.drop(['Label', 'Attack', 'flowStartMilliseconds', 'sourceIPAddress', 'destinationIPAddress', 'sourceTransportPort', 'destinationTransportPort'], 1)
    features = X.columns
    if opt.znorm:
        X = z_scaling(X)
    if opt.minmaxnorm:
        X = minmax_scaling(X)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=data['Attack'])
    X_train_main, X_train_safe, y_train_main, y_train_safe = train_test_split(X, y, test_size=0.5, random_state=SEED, stratify=y)
    del data
    del X
    del y
    return X_train_main, X_train_safe, X_test, y_train_main, y_train_safe, y_test, features

def build_safe_detector_flipper(X, y, M1):

    preds = M1.predict(X).round()
    lines = [i for i,j in enumerate(preds.flatten()) if j!= y[i]]
    Tot = len(features)
    #select 100000 random samples from X_test
    #lines = random.sample(range(X_test.shape[0]), 50000)
    storage = np.ones((len(lines), Tot))

    __min = np.min(X, 0)
    __max = np.max(X, 0)

    ii = 0
    for sample in tqdm(lines):
        label = y[sample]
        for index in range(Tot):

            _min = __min[index]
            _max = __max[index]
            total = _max-_min

            if _min == _max:
                continue

            step = (_max-_min)/opt.step
            right = left = total

            tmp = X[sample].copy()
            start = tmp[index]

            for i in np.arange(start, _max, step):
                tmp[index] = i
                pred = np.round(M1.predict(np.array([tmp]))[0][0])
                if pred == label:
                    right = i-start
                    break

            for i in np.arange(start, _min, -step):
                tmp[index] = i
                pred = np.round(M1.predict(np.array([tmp]))[0][0])
                if pred == label:
                    left = i-start
                    break

            storage[ii, index] = min(right, left)#/(_max-_min) don't normalize stupid
        ii+=1
    return storage, lines

def build_safe_detector_maxconfusion(X, y, M1):

    preds = M1.predict(X)
    confidence_threshold = 0.9

    lines = [i for i,j in enumerate(preds.flatten()) if j == y[i]]
    #lines = [i for i,j in enumerate(pred.flatten()) if (np.abs(pred - 0.5) + 0.5) < confidence_threshold]
    Tot = len(features)
    #select 100000 random samples from X_test
    #lines = random.sample(range(X_test.shape[0]), 50000)
    storage = np.ones((len(lines), Tot))

    __min = np.min(X, 0)
    __max = np.max(X, 0)

    ii = 0
    for sample in tqdm(lines):
        label = y[sample]
        for index in range(Tot):

            _min = __min[index]
            _max = __max[index]
            total = _max-_min

            if _min == _max:
                continue

            step = (_max-_min)/opt.step
            right = left = total

            tmp = X[sample].copy()
            start = tmp[index]

            mem = []
            initial_confidence = np.abs(preds[sample] - 0.5) + 0.5

            for i in np.arange(start, _max, step):
                tmp[index] = i
                pred = np.round(M1.predict(np.array([tmp]))[0][0])
                new_confidence = np.abs(pred - 0.5) + 0.5
                if (pred.round() == label) and (new_confidence < initial_confidence):
                    mem.append((i-start, - new_confidence + initial_confidence))

            for i in np.arange(start, _min, -step):
                tmp[index] = i
                pred = np.round(M1.predict(np.array([tmp]))[0][0])
                new_confidence = np.abs(pred - 0.5) + 0.5
                if (pred.round() == label) and (new_confidence < initial_confidence):
                    mem.append((i-start, - new_confidence + initial_confidence))

            # find the element that has max confidence drop with minimum shift
            best = np.argmax([combination[1]/np.abs(combination[0]) for combination in mem])
            storage[ii, index] = mem[best][0] # store only the shift needed
        ii+=1
    return storage, lines

def SAFE_for_enhancing():
    ####### Train a SAFE detector
    if opt.shifts:
        shifts = np.load(os.path.join(opt.shifts, 'shifts.npy'))
        lines = np.load(os.path.join(opt.shifts, 'lines.npy'))
    else:
        shifts, lines = build_safe_detector_flipper(X_train_safe, y_train_safe, M1)
        np.save(os.path.join(total_path, 'shifts'), shifts)
        np.save(os.path.join(total_path, 'lines'), lines)
    X_train_safe_shifts = get_shifts_for_dataset(X_train_safe, shifts, lines)


    ####### Train a SAFE enhancer
    if opt.kdtree:
        if opt.tree:
            with open(os.path.join(opt.tree, 'kdtree'), 'rb') as tree_object:
                tree = pickle.load(tree_object)
            enhancer = enhancer_kdtree(shifts = X_train_safe_shifts, tree=tree)
        else:
            enhancer = enhancer_kdtree(shifts = X_train_safe_shifts)
            tree = enhancer.make(X_train_safe)
            with open(os.path.join(total_path, 'kdtree'), 'wb') as tree_object:
                pickle.dump(tree, tree_object)

        if opt.mode == "all":
            test_shifts = enhancer.get(X_test, k=opt.k)

        if opt.mode == "only-problematic":
            pred = M1.predict(X_test).flatten()
            confidence = np.abs(pred - 0.5) + 0.5
            candidates = np.argwhere(confidence < 0.9).flatten()
            test_shifts = enhancer.get(X_test[candidates], k=opt.k)
            ####### Test a SAFE enhancer
            y_pred = M1.predict(X_test[candidates])
            a = accuracy_score(y_test[candidates], y_pred.round())
            print('Original Accuracy is:', a*100)

            y_pred = M1.predict(X_test[candidates] + test_shifts)
            a = accuracy_score(y_test[candidates], y_pred.round())
            print('Enhanced Accuracy is:', a*100)
        if opt.mode == "hybrid":
            predictions = []
            for i, sample in enumerate(X_test):
                pred = M1.predict(np.array([sample]))[0][0]
                confidence = np.abs(pred - 0.5) + 0.5
                if confidence < 0.9:
                    sample_shift = enhancer.get(sample, k=opt.k)
                    pred = M1.predict(np.array([sample + sample_shift]))[0][0]
                predictions.append(pred.round())
            y_pred = M1.predict(X_test)
            a = accuracy_score(y_test, y_pred.round())
            print('Original Accuracy is:', a*100)

            a = accuracy_score(y_test, predictions)
            print('Enhanced Accuracy is:', a*100)

    ####### Test a SAFE enhancer
    #y_pred = M1.predict(X_test)
    #a = accuracy_score(y_test, y_pred.round())
    #print('Original Accuracy is:', a*100)

    #y_pred = M1.predict(X_test + test_shifts)
    #a = accuracy_score(y_test, y_pred.round())
    #print('Enhanced Accuracy is:', a*100)

def SAFE_for_robustness_V1():
    ####### Train a SAFE detector
    if opt.shifts:
        shifts = np.load(os.path.join(opt.shifts, 'shifts.npy'))
        lines = np.load(os.path.join(opt.shifts, 'lines.npy'))
    else:
        shifts, lines = build_safe_detector_maxconfusion(X_train_safe, y_train_safe, M1)
        np.save(os.path.join(total_path, 'shifts'), shifts)
        np.save(os.path.join(total_path, 'lines'), lines)
    X_train_safe_shifts = get_shifts_for_dataset(X_train_safe, shifts, lines)

    ####### Train a SAFE enhancer
    if opt.kdtree:
        if opt.tree:
            with open(os.path.join(opt.tree, 'kdtree'), 'rb') as tree_object:
                tree = pickle.load(tree_object)
            enhancer = enhancer_kdtree(shifts = X_train_safe_shifts, tree=tree)
        else:
            enhancer = enhancer_kdtree(shifts = X_train_safe_shifts)
            tree = enhancer.make(X_train_safe)
            with open(os.path.join(total_path, 'kdtree'), 'wb') as tree_object:
                pickle.dump(tree, tree_object)

        if opt.mode == "all":
            new_X_train = np.array((X_train_main))

            #X_train_main, y_train_main

            for i, sample in enumerate(X_train_main):
                new_X_train[i,:] = sample + enhancer.get(sample, k=opt.k)

            new_X_train = np.concatenate((X_train_main, new_X_train))
            new_y_train = np.concatenate((y_train_main, y_train_main))

            new_X_train, new_y_train = unison_shuffled_copies(new_X_train, new_y_train)

            if opt.netrob:
                M2 = load_model(opt.netrob)
            else:
                M2 = build_nn(opt.nLayers, [opt.layerSize for i in range(opt.nLayers-1)], ['sigmoid' for i in range(opt.nLayers)])
                history = M2.fit(new_X_train, new_y_train, validation_data = (X_test, y_test), epochs=opt.nEpoch, batch_size=opt.batchSize)
                M2.save(os.path.join(total_path, 'neural_net_robust.h5'))

    ####### Test a SAFE enhancer
    y_pred = M1.predict(X_test)
    a = accuracy_score(y_test, y_pred.round())
    print('M1 Accuracy is:', a*100)

    y_pred = M2.predict(X_test)
    a = accuracy_score(y_test, y_pred.round())
    print('M2 Accuracy is:', a*100)

def SAFE_for_robustness_V2():
    if opt.shifts:
        shifts = np.load(os.path.join(opt.shifts, 'shifts.npy'))
        lines = np.load(os.path.join(opt.shifts, 'lines.npy'))
    else:
        shifts, lines = build_safe_detector_flipper(X_train_safe, y_train_safe, M1)
        np.save(os.path.join(total_path, 'shifts'), shifts)
        np.save(os.path.join(total_path, 'lines'), lines)
    X_train_safe_shifts = get_shifts_for_dataset(X_train_safe, shifts, lines)

    ####### Train a SAFE enhancer
    if opt.kdtree:
        if opt.tree:
            with open(os.path.join(opt.tree, 'kdtree'), 'rb') as tree_object:
                tree = pickle.load(tree_object)
            enhancer = enhancer_kdtree(shifts = X_train_safe_shifts, tree=tree)
        else:
            enhancer = enhancer_kdtree(shifts = X_train_safe_shifts)
            tree = enhancer.make(X_train_safe)
            with open(os.path.join(total_path, 'kdtree'), 'wb') as tree_object:
                pickle.dump(tree, tree_object)


        if opt.mode == "hybrid":
            # original data
            predictions = []
            for i, sample in enumerate(X_test):
                pred = M1.predict(np.array([sample]))[0][0]
                confidence = np.abs(pred - 0.5) + 0.5
                if confidence < 0.8:
                    sample_shift = enhancer.get(sample, k=opt.k)
                    pred = M1.predict(np.array([sample + sample_shift]))[0][0]
                predictions.append(pred.round())
            y_pred = M1.predict(X_test)
            a = accuracy_score(y_test, y_pred.round())
            print('Original Accuracy is:', a*100)

            a = accuracy_score(y_test, predictions)
            print('Enhanced Accuracy is:', a*100)

            # adversarial data
            classifier_old = KerasClassifier(model=M1)
            attack_old = FastGradientMethod(estimator=classifier_old, eps=0.01)
            X_test_adv = attack_old.generate(x=X_test)

            predictions = []
            for i, sample in enumerate(X_test_adv):
                pred = M1.predict(np.array([sample]))[0][0]
                confidence = np.abs(pred - 0.5) + 0.5
                if confidence < 0.8:
                    sample_shift = enhancer.get(sample, k=opt.k)
                    pred = M1.predict(np.array([sample + sample_shift]))[0][0]
                predictions.append(pred.round())
            y_pred = M1.predict(X_test_adv)
            a = accuracy_score(y_test, y_pred.round())
            print('ADV Original Accuracy is:', a*100)

            a = accuracy_score(y_test, predictions)
            print('ADV Enhanced Accuracy is:', a*100)

def SAFE_for_robustness_V3():
    # generate adversarial data
    classifier_old = KerasClassifier(model=M1)
    attack_old = FastGradientMethod(estimator=classifier_old, eps=0.01)
    X_train_safe_adv = attack_old.generate(x=X_train_safe)

    if opt.shifts:
        shifts = np.load(os.path.join(opt.shifts, 'shifts.npy'))
        lines = np.load(os.path.join(opt.shifts, 'lines.npy'))
    else:
        shifts, lines = build_safe_detector_flipper(X_train_safe_adv, y_train_safe, M1)
        np.save(os.path.join(total_path, 'shifts'), shifts)
        np.save(os.path.join(total_path, 'lines'), lines)
    X_train_safe_shifts = get_shifts_for_dataset(X_train_safe_adv, shifts, lines)

    ####### Train a SAFE enhancer
    if opt.kdtree:
        if opt.tree:
            with open(os.path.join(opt.tree, 'kdtree'), 'rb') as tree_object:
                tree = pickle.load(tree_object)
            enhancer = enhancer_kdtree(shifts = X_train_safe_shifts, tree=tree)
        else:
            enhancer = enhancer_kdtree(shifts = X_train_safe_shifts)
            tree = enhancer.make(X_train_safe_adv)
            with open(os.path.join(total_path, 'kdtree'), 'wb') as tree_object:
                pickle.dump(tree, tree_object)


        if opt.mode == "hybrid":
            # crafted test data
            X_test_adv = attack_old.generate(x=X_test)

            predictions = []
            for i, sample in enumerate(X_test_adv):
                pred = M1.predict(np.array([sample]))[0][0]
                confidence = np.abs(pred - 0.5) + 0.5
                if confidence < 0.9:
                    sample_shift = enhancer.get(sample, k=opt.k)
                    pred = M1.predict(np.array([sample + sample_shift]))[0][0]
                predictions.append(pred.round())
            y_pred = M1.predict(X_test_adv)
            a = accuracy_score(y_test, y_pred.round())
            print('ADV Accuracy is:', a*100)

            a = accuracy_score(y_test, predictions)
            print('Enhanced ADV Accuracy is:', a*100)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_shifts_for_dataset(X, shifts, lines):
    __min = np.min(X, 0)
    __max = np.max(X, 0)
    interval = __max - __min
    X_train_safe_shifts = np.zeros(X.shape)
    for i,j in enumerate(lines):
        tmp = shifts[i]/interval
        X_train_safe_shifts[j, np.argmin(tmp)] = shifts[i, np.argmin(tmp)]
    return X_train_safe_shifts

def SAFE_robustness(shifts, all=True):
    r = np.mean(shifts, 0)

class enhancer_kdtree():
    def __init__(self, tree=None, shifts=None):
        self.tree = tree
        self.shifts = shifts

    def make(self, X_train):
        self.tree = KDTree(X_train)#, metric=DistanceMetric.get_metric('minkowski'))
        return self.tree

    def get(self, X_test, k):
        output = np.zeros(X_test.shape)
        _, ind = self.tree.query(np.array([X_test]), k=k)
        return np.mean(self.shifts[ind.flatten()], 0).flatten()


class enhancer_nn():
    pass

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--nLayers', type=int, default=5)
    parser.add_argument('--layerSize', type=int, default=64)
    parser.add_argument('--nEpoch', type=int, default=32, help='number of epochs to train for')
    parser.add_argument('--net', default='', help="path to pre-trained M1")
    parser.add_argument('--netrob', default='', help="path to pre-trained M2")
    parser.add_argument('--znorm', action='store_true')
    parser.add_argument('--minmaxnorm', action='store_true')
    parser.add_argument('--kdtree', action='store_true')
    parser.add_argument('--tree', default='', help="path to pre-trained kdtree")
    parser.add_argument('--step', type=int, default=10, help='divide the interval into n steps')
    parser.add_argument('--shifts', default='', help="path to shifts")
    parser.add_argument('--k', type=int, default=3, help='n neighbours when using kdtree')
    parser.add_argument('--mode', default='hybrid', help="mode to run (all, only-problematic, hybrid)")
    parser.add_argument('--goal', default='enhancing', help="goal to achieve (enhancing, robustness)")

    opt = parser.parse_args()

    X_train_main, X_train_safe, X_test, y_train_main, y_train_safe, y_test, features = load_data(opt.dataroot)

    ####### Create paths
    main_path = re.search(r'\d+', opt.dataroot).group()
    mode_path = opt.mode
    goal_path = opt.goal
    run_path = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    total_path = os.path.join(main_path, mode_path + '-' + goal_path, run_path)
    if not os.path.exists(total_path):
        os.makedirs(total_path)

    with open(os.path.join(total_path, 'config.cfg'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    ####### Train a simple keras model
    if opt.net:
        M1 = load_model(os.path.join(opt.net, 'neural_net.h5'))
    else:
        M1 = build_nn(opt.nLayers, [opt.layerSize for i in range(opt.nLayers-1)], ['sigmoid' for i in range(opt.nLayers)])
        history = M1.fit(X_train_main, y_train_main, validation_data = (X_test, y_test), epochs=opt.nEpoch, batch_size=opt.batchSize)
        M1.save(os.path.join(total_path, 'neural_net.h5'))

    if opt.goal == 'enhancing':
        SAFE_for_enhancing()

    if opt.goal == 'robustnessV1':
        SAFE_for_robustness_V1()

    if opt.goal == 'robustnessV2':
        SAFE_for_robustness_V2()

    if opt.goal == 'robustnessV3':
        SAFE_for_robustness_V3()
    
