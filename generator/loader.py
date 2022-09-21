import os
import numpy as np
import cv2


class Loader:
  ### root_path: None if the file_path is full path
  def __init__(self, root_path=None):
    self.root_path = root_path

class ImageLoader(Loader):
  def __init__(self, root_path=None, resize=None):
    Loader.__init__(self, root_path)
    self.resize = resize

  def get_data(self, file_path):
    if (self.root_path):
      file_path = os.path.join(self.root_path, file_path)

    data = cv2.imread(file_path).astype(np.float32)
    if (self.resize is not None):
      data = cv2.resize(data, (self.resize[0], self.resize[1]))
    data /= 255.0
    return data

