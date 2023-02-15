import libs.data_operations as data_ops
import os
import logging


def create_folder(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

def checkDecodingWAVCorrectness(filenames):
  for filename in filenames:
    data = data_ops.ReadFile().data_op(filename, 0, 0, 0)
    try:
      data_ops.DecodeWav().data_op(data, 0, 0, 0)
    except Exception as e:
      logging.info(f"error at: {filename} \n {e}")
