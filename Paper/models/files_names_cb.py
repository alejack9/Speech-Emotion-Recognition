from tensorflow.keras.callbacks import Callback

class FirstFilenameSetCallback(Callback):
    def __init__(self, train, val, test):
      self.train_fnames = train
      self.val_fnames = val
      self.test_fnames = test

    def on_train_batch_begin(self, batch, _=None):
      print(f"{batch} train batch:  {self.train_fnames.iloc[batch-1]}")

    def on_test_batch_begin(self, batch, _=None):
      print(f"{batch} val batch:  {self.val_fnames.iloc[batch-1]}")

    def on_predict_batch_begin(self, batch, _=None):
      print(f"{batch} test batch:  {self.test_fnames.iloc[batch-1]}")
