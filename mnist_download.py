import subprocess
import pickle
import gzip
import os

import numpy as np
import matplotlib.pyplot as plt

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}


def downloadZip():
  for v in key_file.values():
    subprocess.call(['wget', url_base + v])


def removeZip():
  for v in key_file.values():
    subprocess.call(['rm', v])


def loadData(file_name, offset):
  with gzip.open(file_name, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=offset)
  return data


def attachNoise(p, data):
  return np.where(
      np.random.uniform(0, 1, data.size).reshape(-1, 1, 28, 28) < p,
      np.random.uniform(0, 1, data.size).reshape(-1, 1, 28, 28), data)


def makeDataset():
  dataset = {}
  dataset['train_img'] = loadData(key_file['train_img'], 16).reshape(-1, 1, 28, 28).astype(
      np.float32) / 255
  dataset['train_label'] = loadData(key_file['train_label'], 8)

  dataset['test_img'] = loadData(key_file['test_img'], 16).reshape(-1, 1, 28, 28).astype(
      np.float32) / 255
  dataset['test_label'] = loadData(key_file['test_label'], 8)
  return dataset


def showExample(data_name, dataset):
  print('train shape')
  print('img   ' + str(dataset['train_img'].shape))
  print('train ' + str(dataset['train_label'].shape))
  print('test shape')
  print('img   ' + str(dataset['test_img'].shape))
  print('train ' + str(dataset['test_label'].shape))
  print('')
  print('train[0] img & label')
  file_name = os.path.join('picture', data_name.replace(' ', '_') + '.png')
  label = str(dataset['train_label'][0])
  print('img   show ' + file_name)
  print('label ' + label)
  example = dataset['train_img'][0].reshape(28, 28)
  plt.title(data_name)
  plt.imshow(example)
  if not os.path.exists('picture'):
    os.makedirs('picture')
  plt.savefig(file_name)


def savePickle(file_name, dataset):
  if not os.path.exists('data'):
    os.makedirs('data')
  with open(os.path.join('data', file_name), 'wb') as f:
    pickle.dump(dataset, f, -1)


if __name__ == '__main__':
  downloadZip()

  dataset = makeDataset()
  showExample('example', dataset)
  savePickle('mnist.pkl', dataset)

  for p in [0.05, 0.10, 0.15, 0.20, 0.25]:
    dataset_noise = dataset.copy()
    dataset_noise['train_img'] = attachNoise(p, dataset_noise['train_img'])
    showExample('example ' + str(int(p * 100)) + '% noise', dataset_noise)
    savePickle('mnist_noise_' + str(int(p * 100)) + '.pkl', dataset_noise)

  removeZip()
