import time

import numpy as np

from model import *
from util import getArgs, getMNIST, plotResult


def genClassifier():
  model = Sequential()
  model.addLayer(ConvolutionLayer(1, 4))
  model.addLayer(ReLULayer())
  model.addLayer(AvgPoolingLayer(4))
  model.addLayer(FlattenLayer())
  model.addLayer(LinearLayer(7 * 7 * 4, 10))
  return Classifier(model)


if __name__ == '__main__':
  args = getArgs()

  trial_name = args.name if args.noise == 0 else args.name + '_' + str(args.noise) + '%_noise'

  classifier = genClassifier()

  train, test = getMNIST(args.noise)

  epoch_time = [0]
  train_acc = []
  test_acc = []
  train_loss = []
  test_loss = []

  batchsize = 100  # 100
  n_train = 60000  # 60000
  n_test = 10000  # 10000
  epoch = 5

  for e in range(1, epoch + 1):
    epoch_start = time.time()
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
    epoch_time.append(epoch_time[-1] + time.time() - epoch_start)

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

  plotResult(trial_name, epoch, (train_acc, train_loss), (test_acc, test_loss), epoch_time[1:])
