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

  def extract_mfcc(self, pcm):
    # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    pcm = tf.convert_to_tensor(value=pcm, dtype=tf.float32)
    stfts = tf.signal.stft(pcm, frame_length=self.win_length, frame_step=self.hop_step, fft_length=self.fft_length)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1].value
    lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins,
                                                                        num_spectrogram_bins,
                                                                        self.sample_rate,
                                                                        lower_edge_hertz,
                                                                        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, axes=[[2], [0]])
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    return log_mel_spectrograms

  def ear_compute(self, landmarks):
    ears = []
    for ps in landmarks:
      ps = list(map(lambda x: float(x), ps))

      EAR1 = float((math.sqrt((ps[74] - ps[82]) ** 2 + (ps[75] - ps[83]) ** 2) + math.sqrt(
          (ps[76] - ps[80]) ** 2 + (ps[77] - ps[81]) ** 2))) / math.sqrt(
          (ps[72] - ps[78]) ** 2 + (ps[73] - ps[79]) ** 2)
      EAR2 = float((math.sqrt((ps[86] - ps[94]) ** 2 + (ps[87] - ps[95]) ** 2) + math.sqrt(
          (ps[88] - ps[92]) ** 2 + (ps[89] - ps[93]) ** 2))) / math.sqrt(
          (ps[84] - ps[90]) ** 2 + (ps[85] - ps[91]) ** 2)
      EAR = (EAR1 + EAR2) / 2
      ears.append([EAR])

    return np.array(ears)

  def split_bfmcoeff(self, coeff):
    id_coeff = coeff[:80]  # identity(shape) coeff of dim 80
    ex_coeff = coeff[80:144]  # expression coeff of dim 64
    tex_coeff = coeff[144:224]  # texture(albedo) coeff of dim 80
    angle = coeff[224:227]  # ruler angle(x,y,z) for rotation of dim 3
    gamma = coeff[227:254]  # lighting coeff for 3 channel SH function of dim 27
    translation = coeff[254:]  # translation coeff of dim 3

    return id_coeff, ex_coeff, tex_coeff, angle, gamma, translation

  def pose_compute(self, bfmcoeffs):
    poses = []
    for bfmcoeff in bfmcoeffs:
      _, _, _, angle, _, _ = self.split_bfmcoeff(bfmcoeff)
      poses.append(angle)

    return np.array(poses)


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

    # example_img = image_loader.get_data('/media/dong/DiskData/gridcorpus/todir_vid2vid/vid1/05/1.jpg')
    # example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
    # example_img = np.concatenate([example_img[:, :self.img_size, :], 
    #                               example_img[:, self.img_size:self.img_size*2, :], 
    #                               example_img[:, self.img_size*2:, :]], 
    #                               axis=-1)
    # example_img = np.concatenate([example_img[:, :, :3], 
    #                               example_img[:, :, 3:6], 
    #                               example_img[:, :, 6:]], 
    #                               axis=1)

    random.shuffle(self.data_list)

    for line in self.data_list:
      folder, img_count = line.strip().split('|')
      img_count = int(img_count)

      for i in range(img_count):
        select_idx = random.choice([j for j in [i-5, i-4, i-3, i-2, i-1, i+1, i+2, i+3, i+4, i+5] if j>=0 and j<img_count])
        rsize = random.randint(int(self.img_size*self.crop_ratio), self.img_size)
        rx = random.randint(0, self.img_size - rsize)
        ry = random.randint(0, self.img_size - rsize)

        example_img = image_loader.get_data(os.path.join(folder, '{:04d}.png'.format(select_idx)))
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

        # rsize = random.randint(int(self.img_size*self.crop_ratio), self.img_size)
        # rx = random.randint(0, self.img_size - rsize)
        # ry = random.randint(0, self.img_size - rsize)

        img = image_loader.get_data(os.path.join(folder, '{:04d}.png'.format(i)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.concatenate([img[:, :self.img_size, :], img[:, self.img_size:self.img_size*2, :], img[:, self.img_size*2:, :]], axis=-1)
        img = img[rx:rsize+rx, ry:rsize+ry, :]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.concatenate([img[:, :, :3], img[:, :, 3:6], img[:, :, 6:]], axis=1)

        imgs = []
        imgs.append(example_img)
        imgs.append(img)
        imgs = np.array(imgs)

        inputs = imgs[:, :, self.img_size:self.img_size*2, :]
        inputs = inputs.transpose((1, 2, 0, 3))
        inputs = inputs.reshape([self.img_size, self.img_size, 6])
        targets = imgs[:, :, :self.img_size, :]
        masks = imgs[:, :, self.img_size*2:, :]
        fg_inputs = targets * masks
        fg_inputs = fg_inputs.transpose([1, 2, 0, 3]).reshape([self.img_size, self.img_size, 6])
        # fg_inputs = targets[0, ...] * masks[0, ...]
        yield inputs, fg_inputs, targets[1, ...], masks[1, ...]

  def get_dataset(self):
    self.set_params(self.__params)

    dataset = tf.data.Dataset.from_generator(
        self.iterator,
        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=([self.img_size, self.img_size, 6], 
                        [self.img_size, self.img_size, 6], 
                        [self.img_size, self.img_size, 3], 
                        [self.img_size, self.img_size, 3])
    )

    dataset = dataset.shuffle(self.shuffle_bufsize).repeat()
    dataset = dataset.padded_batch(self.batch_size,
                                   padded_shapes=([self.img_size, self.img_size, 6], 
                                                  [self.img_size, self.img_size, 6], 
                                                  [self.img_size, self.img_size, 3], 
                                                  [self.img_size, self.img_size, 3]))

    return dataset
