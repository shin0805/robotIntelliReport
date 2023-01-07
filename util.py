import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle


def softmax(y):
  y = y - np.max(y, axis=1, keepdims=True)
  return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)


def cross_entropy(prob, t):
  return -np.mean(np.log(prob[np.arange(prob.shape[0]), t] + 1e-7))


def softmax_cross_entropy(y, t):
  return cross_entropy(softmax(y), t)


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


def plotAccuracy(name, epoch, data):
  plt.title(name + ' accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.xlim([1, epoch])
  plt.ylim([0, 1])
  plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
  plt.plot(np.linspace(1, epoch, len(data)), data, 'r-')
  plt.axhline(y=data[-1], color='black', linestyle='--')
  plt.text(1.01, data[-1] + 0.01, str(data[-1]))
  if not os.path.exists('picture'):
    os.makedirs('picture')
  plt.savefig(os.path.join('picture', name + '_accuracy.jpg'))
  plt.clf()