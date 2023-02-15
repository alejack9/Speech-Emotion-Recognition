from tensorflow.keras.callbacks import Callback

class LastEpochWriterCallback(Callback):
    def __init__(self, filename):
      self.filename = filename

    def on_epoch_end(self, epoch, _=None):
      with open(self.filename, "w+") as f:
          f.write(f"{epoch}")
