import os
import csv
import argparse

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


def getArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', '-nm', default='example', help='the name of trial')
  parser.add_argument('--noise', '-ns', default=0, type=int, help='the probability of noise [%%]')
  return parser.parse_args()


def getMNIST(noise):
  train_img = np.zeros((60000, 1 * 28 * 28), dtype=np.float32)
  train_label = np.zeros(60000, dtype=np.int32)
  test_img = np.zeros((10000, 1 * 28 * 28), dtype=np.float32)
  test_label = np.zeros(10000, dtype=np.int32)

  if noise == 0:
    dataset = pickle.load(open(os.path.join('data', 'mnist.pkl')))
  else:
    dataset = pickle.load(open(os.path.join('data', 'mnist_noise_' + str(noise) + '.pkl')))

  train_img = dataset['train_img']
  train_label = dataset['train_label']
  test_img = dataset['test_img']
  test_label = dataset['test_label']
  return (train_img, train_label), (test_img, test_label)


def saveResult(name, train, test, time):
  path = os.path.join('result', 'result.csv')
  if not os.path.exists(path):
    with open(path, 'a') as f:
      writer = csv.writer(f)
      writer.writerows([['trial name', 'train acc', 'train loss', 'test acc', 'test loss',
                         'time']])
  with open(path, 'a') as f:
    writer = csv.writer(f)
    writer.writerows([[name.replace('_', ' '), train[0], train[1], test[0], test[1], time]])


def plotResult(name, epoch, train, test, time):
  if not os.path.exists(os.path.join('result', name)):
    os.makedirs(os.path.join('result', name))
  saveResult(name, (train[0][-1], train[1][-1]), (test[0][-1], test[1][-1]), time[-1])
  plotAccuracy(name, epoch, train[0])
  plotAccuracy(name, epoch, test[0])
  plotLoss(name, epoch, train[1])
  plotLoss(name, epoch, test[1])
  plotTime(name, epoch, time)


def plotAccuracy(name, epoch, data):
  plt.title(('test' if epoch == len(data) else 'train') + ' accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.xlim([1, epoch])
  plt.ylim([0, 1])
  plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
  plt.plot(np.linspace(1, epoch, len(data)), data, 'ro-' if epoch == len(data) else 'r-')
  plt.axhline(y=data[-1], color='black', linestyle='--')
  plt.text(1.01, data[-1] + 0.01, str(data[-1]))
  plt.savefig(
      os.path.join('result', name, ('test' if epoch == len(data) else 'train') + '_accuracy.jpg'))
  plt.clf()


def plotLoss(name, epoch, data):
  plt.title(('test' if epoch == len(data) else 'train') + ' loss')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.xlim([1, epoch])
  plt.ylim([0, 2.5])
  plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
  plt.plot(np.linspace(1, epoch, len(data)), data, 'bo-' if epoch == len(data) else 'b-')
  plt.axhline(y=data[-1], color='black', linestyle='--')
  plt.text(1.01, data[-1] + 0.01, str(data[-1]))
  plt.savefig(
      os.path.join('result', name, ('test' if epoch == len(data) else 'train') + '_loss.jpg'))
  plt.clf()


def plotTime(name, epoch, data):
  plt.title('elapsed time')
  plt.xlabel('epoch')
  plt.ylabel('time / s')
  plt.xlim([1, epoch])
  plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
  plt.plot(np.linspace(1, epoch, len(data)), data, 'ko-')
  plt.axhline(y=data[-1], color='black', linestyle='--')
  plt.text(1.01, data[-1] + 0.01, str(data[-1]))
  plt.savefig(os.path.join('result', name, 'time.jpg'))
  plt.clf()
