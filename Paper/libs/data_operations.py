import abc
import tensorflow as tf
import tensorflow_io as tfio
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DataOperation(abc.ABC):
    def __init__(self):
        pass

    def data_op(self, data: tf.Tensor, start, end):
        """Should never be called directly."""
        raise NotImplementedError()

    def __call__(self, data: tf.Tensor, label: tf.Tensor, start, end):
        return self.data_op(data, start, end), label, start, end

class RemoveCropInformation():
    def __call__(self, data: tf.Tensor, label: tf.Tensor, start, end):
        return data, label

class Log(DataOperation):
    def __init__(self, after):
        self.after = after
        super().__init__()
    def data_op(self, data: tf.Tensor, start, end):
        logging.info(f"After '{self.after}':")
        logging.info(data)
        logging.info('---------------------')
        return data

class ReadFile(DataOperation):
    def data_op(self, data: tf.Tensor, start, end):
        """
        `data operation` version of tf io read_file
        :param data: Tensor with filename, dtype=string
        :return: Tensor with file contents, dtype=string
        """
        return tf.io.read_file(data)

class Resample(DataOperation):
    def __init__(self, rate_out):
        self.rate_out = rate_out
        super().__init__()

    def data_op(self, data, start, end):
        """
        Resamples the passed data (first element in the tuple) assuming that the second element is the input rate
        :param data: Tensor with dtype=string || (Tensor with dtype=string, int64)
        :return: Tensor with file contents, dtype=string
        """
        if type(data) is tuple:
            return tfio.audio.resample(data[0], data[1], self.rate_out)
        else:
            return data

class DecodeWav(DataOperation):
    def __init__(self, with_sample_rate: bool):
        self.with_sample_rate = with_sample_rate
        super().__init__()

    def data_op(self, data: tf.Tensor, start, end):
        """
        Decode a wave file which was read by `tf.io.read_file`. If `with_sample_rate` is true, also returns the audio's sample rate
        :param data: Tensor with dtype=string
        :return: Tensor with dtype=string || (Tensor with dtype=string, int64)
        """
        if not self.with_sample_rate:
            return tfio.audio.decode_wav(data, dtype=tf.int16)
        else:
            return tfio.audio.decode_wav(data, dtype=tf.int16), tf.cast(tf.audio.decode_wav(data)[1], tf.int64)


class Squeeze(DataOperation):
    def data_op(self, data: tf.Tensor, start, end):
        """
        Drop the last dimension of a Tensor
        :param data: Tensor with dimension [x, 1]
        :return: Tensor with dimension [x, ]
        """
        return tf.squeeze(data, -1)


class Crop(DataOperation):
    def data_op(self, data: tf.Tensor, start, end):
        """
        Crop signal to length
        :param data:
        :return:
        """
        min = tf.math.minimum(end, tf.size(data))
        return tf.slice(data, [start], [min - start])


class ZeroPad(DataOperation):
    def __init__(self, length=64_000):
        self.length = length
        super().__init__()

    def data_op(self, data: tf.Tensor, start, end):
        """
        Add zeros to a 1D tensor to match given length
        :param data: 1D numerical tensor of length below given length
        :return: 1D numerical tensor of given length
        """
        return tf.concat([data, tf.zeros(self.length - tf.shape(data), dtype=tf.int16)], axis=0)

class CastToFloat(DataOperation):
    def __init__(self):
        super().__init__()

    def data_op(self, data: tf.Tensor, start, end):
        return tf.cast(data, tf.float32)

class Normalize(DataOperation):
    def __init__(self):
        super().__init__()

    def data_op(self, data: tf.Tensor, start, end):
        # 32768 is the max amplitude of wav signals 
        return data / 32768.0
class Reshape(DataOperation):
    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def data_op(self, data: tf.Tensor, start, end):
        return tf.reshape(data, self.shape)
