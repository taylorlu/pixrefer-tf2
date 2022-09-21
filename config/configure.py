#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import yaml
from tensor2tensor.utils.hparam import HParams


class YParams(HParams):
  def __init__(self, yaml_fn, config_name):
    HParams.__init__(self)
    with open(yaml_fn) as fp:
      for k, v in yaml.load(fp, Loader=yaml.FullLoader)[config_name].items():
        self.add_hparam(k, v)
