import time
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from model import *
from util import getMNIST, plotAccuracy

if __name__ == '__main__':
  model = Sequential()
  model.addLayer(ConvolutionLayer(1, 16))
  model.addLayer(ReLULayer())
  model.addLayer(ConvolutionLayer(16, 16))
  model.addLayer(ReLULayer())
  model.addLayer(MaxPoolingLayer(2))
  model.addLayer(ReLULayer())
  model.addLayer(ConvolutionLayer(16, 32))
  model.addLayer(ReLULayer())
  model.addLayer(ConvolutionLayer(32, 32))
  model.addLayer(ReLULayer())
  model.addLayer(AvgPoolingLayer(2))
  model.addLayer(FlattenLayer())
  model.addLayer(LinearLayer(7 * 7 * 32, 10))
  classifier = Classifier(model)

  train, test = getMNIST('mnist.pkl')

  train_acc = []
  test_acc = []
  train_loss = []
  test_loss = []

  batchsize = 100  # 100
  n_train = 600  # 60000
  n_test = 100  # 10000
  epoch = 5

  for e in range(1, epoch + 1):
    randinds = np.random.permutation(n_train)
    for it in range(0, n_train, batchsize):
      ind = randinds[it:it + batchsize]
      x = train[0][ind]
      t = train[1][ind]
      start = time.time()
      loss, acc = classifier.update(x, t)
      end = time.time()
      train_acc.append(acc)
      train_loss.append(loss)
      print('[train] epoch %d, iteration %d, elapsed time %f, loss %f, acc %f' %
            (e, it // batchsize, end - start, loss, acc))

    start = time.time()
    acc_ave = 0
    loss_ave = 0
    for it in range(0, n_test, batchsize):
      x = test[0][it:it + batchsize]
      t = test[1][it:it + batchsize]
      loss, acc = classifier.predict(x, t)
      acc_ave += int(acc * batchsize)
      loss_ave += loss
    acc_ave /= (1.0 * n_test)
    loss_ave /= (n_test // batchsize)
    end = time.time()
    test_acc.append(acc_ave)
    test_loss.append(loss_ave)
    print('[test] epoch %d, elapsed time %f, loss %f, acc %f' %
          (e, end - start, loss_ave, acc_ave))

  plotAccuracy('mini_train', epoch, train_acc)
  plotAccuracy('mini_test', epoch, test_acc)
