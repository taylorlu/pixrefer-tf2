#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from optparse import OptionParser
import logging
import subprocess
from pixrefer import PixReferNet
from generator.loader import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if (__name__ == '__main__'):

  cmd_parser = OptionParser(usage="usage: %prog [options] --config_path <>")
  cmd_parser.add_option('--config_path', type="string", default='config/params.yml', dest="config_path",
                        help='the config yaml file')

  opts, argv = cmd_parser.parse_args()

  if (opts.config_path is None):
    logger.error('Please check your parameters.')
    exit(0)

  config_path = opts.config_path

  if (not os.path.exists(config_path)):
    logger.error('config_path not exists')
    exit(0)

  os.environ["CUDA_VISIBLE_DEVICES"] = '0'
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

  os.makedirs('output', exist_ok=True)
  for file in os.listdir('output'):
    os.system('rm -rf output/{}'.format(file))

  batch_size = 1
  img_size = 512
  image_loader = ImageLoader()
  bg_img = np.zeros([img_size, img_size, 3]).astype(np.float32)

  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))) as sess:
    with tf.compat.v1.variable_scope('recognition'):
      ### Vid2VidNet setting
      vid2vidnet = PixReferNet(config_path)
      params = vid2vidnet.params
      params.batch_size = 1
      params.seq_len = 5
      params.add_hparam('is_training', False)
      vid2vidnet.set_params(params)

      inputs_holder = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size, img_size, 6])
      fg_inputs_holder = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
      targets_holder = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
      vid2vid_nodes = vid2vidnet.build_inference_op(inputs_holder, fg_inputs_holder, targets_holder)

    variables_to_restore = tf.compat.v1.global_variables()
    rec_varlist = {v.name[12:][:-2]: v 
                            for v in variables_to_restore if v.name[:11]=='recognition'}

    rec_saver = tf.compat.v1.train.Saver(var_list=rec_varlist)

    sess.run(tf.compat.v1.global_variables_initializer())
    rec_saver.restore(sess, 'ckpt_pixrefer-2009/pixrefernet-130000')

    inputs = np.zeros([1, img_size, img_size, 6], dtype=np.float32)
    fg_inputs = np.zeros([1, img_size, img_size, 3], dtype=np.float32)

    img = image_loader.get_data('0000.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs[0, :, :, 0:3] = img[:, img_size:img_size*2, :]
    fg_inputs[0, :, :, 0:3] = img[:, :img_size, :] * img[:, img_size*2:, :]

    root = r'/mnt/workplace/DECA/TestVideo/results'
    for index in range(0, len(os.listdir(root))):
      img = image_loader.get_data(os.path.join(root, '{:04d}'.format(index), 'orig_{:04d}_shape_images.jpg'.format(index)))
      if (img is not None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs[0, ..., 3:6] = img

        frames, last, alpha = sess.run([vid2vid_nodes['Outputs'], vid2vid_nodes['Outputs_FG'], vid2vid_nodes['Alphas']], 
          feed_dict={inputs_holder: inputs, fg_inputs_holder: fg_inputs, targets_holder: bg_img[np.newaxis, ...]})

        jpg = cv2.cvtColor((frames[0, ...]*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        alpha = cv2.cvtColor((alpha[0, ...]*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgba = np.concatenate([jpg, alpha[..., :1]], axis=-1)
        cv2.imwrite('output/{:04d}.png'.format(index), rgba)

        inputs[0, ..., :3] = img
        fg_inputs[0, :, :, 0:3] = frames[0, ...]