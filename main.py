import time

import numpy as np

from layer import *
from util import getMNIST

def main():
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

    batchsize = 100
    n_train = 60000
    n_test = 10000
    epoch = 5

    for e in range(epoch):
        print('epoch %d'%e)
        randinds = np.random.permutation(n_train)
        for it in range(0, n_train, batchsize):
            ind = randinds[it:it+batchsize]
            x = train[0][ind]
            t = train[1][ind]
            start = time.time()
            loss, acc = classifier.update(x, t)
            end = time.time()
            print('train iteration %d, elapsed time %f, loss %f, acc %f'%(it//batchsize, end-start, loss, acc))

        start = time.time()
        acctest = 0
        losstest = 0
        for it in range(0, n_test, batchsize):
            x = test[0][it:it+batchsize]
            t = test[1][it:it+batchsize]
            loss, acc = classifier.predict(x, t)
            acctest += int(acc * batchsize)
            losstest += loss
        acctest /= (1.0 * n_test)
        losstest /= (n_test // batchsize)
        end = time.time()
        print('test, elapsed time %f, loss %f, acc %f'%(end-start, loss, acc))

if __name__ == '__main__':
    main()
