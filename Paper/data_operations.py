import abc
import tensorflow as tf
import tensorflow_io as tfio


class DataOperation(abc.ABC):
    def __init__(self):
        pass

    def data_op(self, data: tf.Tensor):
        """Should never be called directly."""
        raise NotImplementedError()

    def __call__(self, data: tf.Tensor, label: tf.Tensor):
        return self.data_op(data), label
    # def __call__(self, data: tf.Tensor):
    #     return self.data_op(data)


class ReadFile(DataOperation):
    def data_op(self, data: tf.Tensor):
        """
        `data operation` version of tf io read_file
        :param data: Tensor with filename, dtype=string
        :return: Tensor with file contents, dtype=string
        """
        file_content = tf.io.read_file(data)
        return file_content


class DecodeWav(DataOperation):
    def data_op(self, data: tf.Tensor):
        """
        Decode a wave file which was read by `tf.io.read_file`
        :param data: Tensor with dtype=string
        :return: Tensor with dtype=int16
        """
        decoded = tfio.audio.decode_wav(data, dtype=tf.int16)
        return decoded


class Squeeze(DataOperation):
    def data_op(self, data: tf.Tensor):
        """
        Drop the last dimension of a Tensor
        :param data: Tensor with dimension [x, 1]
        :return: Tensor with dimension [x, ]
        """
        squeezed = tf.squeeze(data, -1)
        return squeezed


class Crop(DataOperation):
    def __init__(self, length):
        self.length = tf.constant([length])
        super().__init__()

    def data_op(self, data: tf.Tensor):
        """
        Crop signal to length
        :param data:
        :return:
        """
        return tf.slice(data, [0], tf.math.minimum(self.length, tf.size(data)))


class ZeroPad(DataOperation):
    def __init__(self, length=64_000):
        self.length = length
        super().__init__()

    def data_op(self, data: tf.Tensor):
        """
        Add zeros to a 1D tensor to match given length
        :param data: 1D numerical tensor of length below given length
        :return: 1D numerical tensor of given length
        """
        zeros = tf.zeros(self.length - tf.shape(data), dtype=tf.int16)
        padded = tf.concat([data, zeros], axis=0)
        return padded


class CastToFloat(DataOperation):
    def __init__(self):
        super().__init__()

    def data_op(self, data: tf.Tensor):
        floated = tf.cast(data, tf.float32)
        return floated


class Reshape(DataOperation):
    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def data_op(self, data: tf.Tensor):
        return tf.reshape(data, (self.shape,))
