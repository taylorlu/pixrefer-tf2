import tensorflow as tf
import os
import numpy as np
import random
import logging
import sys
import math

sys.path.append(os.path.join(os.getcwd(), 'config'))
sys.path.append(os.path.join(os.getcwd(), 'generator'))

from configure import YParams
from loader import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataGenerator:
  def __init__(self, config_path):
    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = DataGenerator.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.sample_rate = params.mel['sample_rate']
    self.num_mel_bins = params.mel['num_mel_bins']
    self.win_length = params.mel['win_length']
    self.hop_step = params.mel['hop_step']
    self.fft_length = params.mel['fft_length']
    self.frame_rate = params.frame_rate
    self.frame_wav_scale = self.sample_rate / self.frame_rate
    self.frame_mfcc_scale = self.frame_wav_scale / self.hop_step

    assert (self.frame_mfcc_scale - int(self.frame_mfcc_scale) == 0), "sample_rate/hop_step must divided by frame_rate."

    self.frame_mfcc_scale = int(self.frame_mfcc_scale)

  def iterator(self):
    raise NotImplementError('iterator not implemented.')

  def get_dataset(self):
    raise NotImplementError('get_dataset not implemented.')


class PixReferDataGenerator(DataGenerator):
  def __init__(self, config_path):

    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = PixReferDataGenerator.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('dataset_path', params.train_dataset_path)
    params.add_hparam('shuffle_bufsize', 100)
    params.add_hparam('batch_size', 2)
    params.add_hparam('img_size', 512)
    params.add_hparam('crop_ratio', 0.9)
    params.add_hparam('seq_len', 8)
    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.data_list = open(params.dataset_path).readlines()
    self.shuffle_bufsize = params.shuffle_bufsize
    self.batch_size = params.batch_size
    self.img_size = params.img_size
    self.crop_ratio = params.crop_ratio
    self.seq_len = params.seq_len

  def iterator(self):
    image_loader = ImageLoader()

    random.shuffle(self.data_list)

    for line in self.data_list:
      folder, img_count = line.strip().split('|')
      img_count = int(img_count)
      slice = 4

      for i in range(slice-1, img_count-slice+1):
        fwd_bwd = random.choice([-1, 1])
        rsize = random.randint(int(self.img_size*self.crop_ratio), self.img_size)
        rx = random.randint(0, self.img_size - rsize)
        ry = random.randint(0, self.img_size - rsize)

        example_img = image_loader.get_data(os.path.join(folder, '{:04d}.png'.format(i+fwd_bwd*3)))
        example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
        example_img = np.concatenate([example_img[:, :self.img_size, :], 
                                      example_img[:, self.img_size:self.img_size*2, :], 
                                      example_img[:, self.img_size*2:, :]], 
                                      axis=-1)
        example_img = example_img[rx:rsize+rx, ry:rsize+ry, :]
        example_img = cv2.resize(example_img, (self.img_size, self.img_size))
        example_img = np.concatenate([example_img[:, :, :3], 
                                      example_img[:, :, 3:6], 
                                      example_img[:, :, 6:]], 
                                      axis=1)

        imgs = []
        imgs.append(example_img)
        for k in range(2, -1, -1):
          img = image_loader.get_data(os.path.join(folder, '{:04d}.png'.format(i+fwd_bwd*k)))
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          img = np.concatenate([img[:, :self.img_size, :], img[:, self.img_size:self.img_size*2, :], img[:, self.img_size*2:, :]], axis=-1)
          img = img[rx:rsize+rx, ry:rsize+ry, :]
          img = cv2.resize(img, (self.img_size, self.img_size))
          img = np.concatenate([img[:, :, :3], img[:, :, 3:6], img[:, :, 6:]], axis=1)
          imgs.append(img)
        imgs = np.array(imgs)

        inputs = imgs[:, :, self.img_size:self.img_size*2, :]
        inputs = inputs.transpose((1, 2, 0, 3))
        inputs = inputs.reshape([self.img_size, self.img_size, 12])
        targets = imgs[:, :, :self.img_size, :]

        yield inputs, targets.transpose([1,2,0,3]).reshape([self.img_size, self.img_size, 12])

  def get_dataset(self):
    self.set_params(self.__params)

    dataset = tf.data.Dataset.from_generator(
        self.iterator,
        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=([self.img_size, self.img_size, 12], 
                        [self.img_size, self.img_size, 12])
    )

    dataset = dataset.shuffle(self.shuffle_bufsize).repeat()
    dataset = dataset.padded_batch(self.batch_size,
                                   padded_shapes=([self.img_size, self.img_size, 12], 
                                                  [self.img_size, self.img_size, 12]))

    return dataset
