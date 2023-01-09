import time
import sys

import numpy as np

from model import *
from util import getArgs, getMNIST, plotResult


class Runner():

  def __init__(self):
    self.args = getArgs()
    self.trial_name = self.args.name if self.args.noise == 0 else self.args.name + '_' + str(
        self.args.noise) + '%_noise'
    self.model = self.genModel()
    self.classifier = Classifier(self.model)
    self.train, self.test = getMNIST(self.args.noise)
    self.batchsize = 100
    self.n_train = 60000
    self.n_test = 10000
    self.epoch = 5

    self.train_acc = []
    self.test_acc = []
    self.train_loss = []
    self.test_loss = []
    self.epoch_time = [0]

  def genModel(self):
    # Simple Model
    # model = Sequential()
    # model.addLayer(ConvolutionLayer(1, 4))
    # model.addLayer(ReLULayer())
    # model.addLayer(AvgPoolingLayer(4))
    # model.addLayer(FlattenLayer())
    # model.addLayer(LinearLayer(7 * 7 * 4, 10))
    # return model

    # MoreLayer
    model = Sequential()
    model.addLayer(ConvolutionLayer(1, 4))
    model.addLayer(ReLULayer())
    model.addLayer(ConvolutionLayer(4, 4))
    model.addLayer(ReLULayer())
    model.addLayer(MaxPoolingLayer(2))
    model.addLayer(ReLULayer())
    model.addLayer(ConvolutionLayer(4, 8))
    model.addLayer(ReLULayer())
    model.addLayer(ConvolutionLayer(8, 8))
    model.addLayer(ReLULayer())
    model.addLayer(AvgPoolingLayer(2))
    model.addLayer(FlattenLayer())
    model.addLayer(LinearLayer(7 * 7 * 8, 10))
    return model

  def training(self):
    start = time.time()
    rand_ids = np.random.permutation(self.n_train)
    for it in range(0, self.n_train, self.batchsize):
      ids = rand_ids[it:it + self.batchsize]
      it_start = time.time()
      loss, acc = self.classifier.update(self.train[0][ids], self.train[1][ids])
      sys.stdout.write('\r' + '[train] elapsed time %f, loss %f, acc %f (%d%%)' %
                       (time.time() - it_start, loss, acc, 100 * (it // self.batchsize) /
                        ((self.n_train - 1) // self.batchsize)))
      sys.stdout.flush()
      self.train_acc.append(acc)
      self.train_loss.append(loss)
    self.epoch_time.append(self.epoch_time[-1] + time.time() - start)

  def inference(self):
    start = time.time()
    acc_ave = 0
    loss_ave = 0
    for it in range(0, self.n_test, self.batchsize):
      loss, acc = self.classifier.predict(self.test[0][it:it + self.batchsize],
                                          self.test[1][it:it + self.batchsize])
      acc_ave += acc * self.batchsize
      loss_ave += loss
    acc_ave /= self.n_test
    loss_ave /= (self.n_test // self.batchsize)
    print('\n[test]  elapsed time %f, loss %f, acc %f' % (time.time() - start, loss_ave, acc_ave))
    self.test_acc.append(acc_ave)
    self.test_loss.append(loss_ave)

  def run(self):
    for e in range(1, self.epoch + 1):
      print('epoch %d elapsed time %f' % (e, self.epoch_time[e - 1]))
      self.training()
      self.inference()


if __name__ == '__main__':
  runner = Runner()
  runner.run()

  plotResult(runner.trial_name, runner.epoch, (runner.train_acc, runner.train_loss),
             (runner.test_acc, runner.test_loss), runner.epoch_time[1:])
