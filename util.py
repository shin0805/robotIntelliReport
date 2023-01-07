import os
import numpy as np
import pickle

def getMNIST(file_name):
    train_img = np.zeros((60000, 1 * 28 * 28), dtype=np.float32)
    train_label = np.zeros(60000, dtype=np.int32)
    test_img = np.zeros((10000, 1 * 28 * 28), dtype=np.float32)
    test_label = np.zeros(10000, dtype=np.int32)

    dataset = pickle.load(open(os.path.join('data', file_name)))

    train_img = dataset['train_img']
    train_label = dataset['train_label']
    test_img = dataset['test_img']
    test_label = dataset['test_label']
    return (train_img, train_label), (test_img, test_label)
