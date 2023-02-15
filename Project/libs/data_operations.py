import abc
import tensorflow as tf
import tensorflow_io as tfio
import logging

class DataOperation(abc.ABC):
    def __init__(self):
        pass

    def data_op(self, data: tf.Tensor, sample_rate, start, end):
        """Should never be called directly."""
        raise NotImplementedError()

    def __call__(self, data: tf.Tensor, label: tf.Tensor, sample_rate, start, end):
        return self.data_op(data, sample_rate, start, end), label, sample_rate, start, end

class GetSampleAndLabel():
    def __call__(self, data: tf.Tensor, label: tf.Tensor, _, __, ___):
        return data, label

class Log(DataOperation):
    def __init__(self, after):
        self.after = after
        super().__init__()
    def data_op(self, data: tf.Tensor, _, __, ___):
        logging.info(f"After '{self.after}':")
        logging.info(data)
        logging.info('---------------------')
        return data

class ReadFile(DataOperation):
    def data_op(self, data: tf.Tensor, _, __, ___):
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

    def data_op(self, data, sample_rate, _, __):
        """
        Resamples the passed data
        :param data: Tensor with dtype=string
        :return: Tensor with file contents, dtype=string
        """
        return tfio.audio.resample(data, sample_rate, self.rate_out)

class DecodeWav(DataOperation):
    def data_op(self, data: tf.Tensor, _, __, ___):
        """
        Decode a wave file which was read by `tf.io.read_file`
        :param data: Tensor with dtype=string
        :return: Tensor with dtype=string || (Tensor with dtype=string, int64)
        """
        return tfio.audio.decode_wav(data, dtype=tf.int16)

class Squeeze(DataOperation):
    def data_op(self, data: tf.Tensor, _, __, ___):
        """
        Drop the last dimension of a Tensor
        :param data: Tensor with dimension [x, 1]
        :return: Tensor with dimension [x, ]
        """
        return tf.squeeze(data, -1)

class Crop(DataOperation):
    def data_op(self, data: tf.Tensor, _, start, end):
        """
        Crop signal to length
        :param data:
        :return:
        """
        min = tf.math.minimum(end, tf.size(data))
        return tf.slice(data, [start], [min - start])

class ZeroPad(DataOperation):
    def __init__(self, length):
        self.length = length
        super().__init__()

    def data_op(self, data: tf.Tensor, _, __, ___):
        """
        Add zeros to a 1D tensor to match given length
        :param data: 1D numerical tensor of length below given length
        :return: 1D numerical tensor of given length
        """
        return tf.concat([data, tf.zeros(self.length - tf.shape(data), dtype=data.dtype)], axis=0)

class CastToFloat(DataOperation):
    def data_op(self, data: tf.Tensor, _, __, ___):
        return tf.cast(data, tf.float32)

class Normalize(DataOperation):
    def data_op(self, data: tf.Tensor, _, __, ___):
        # 32768 is the max amplitude of wav signals 
        return data / 32768.0

class Reshape(DataOperation):
    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def data_op(self, data: tf.Tensor, _, __, ___):
        return tf.reshape(data, self.shape)

class Fade(DataOperation):
    def __init__(self, fade_in, fade_out):
        self.fade_in = fade_in
        self.fade_out = fade_out
        super().__init__()
    
    def data_op(self, data: tf.Tensor, _, __, ___):
        return tfio.audio.fade(data, fade_in=self.fade_in, fade_out=self.fade_out, mode="logarithmic")
