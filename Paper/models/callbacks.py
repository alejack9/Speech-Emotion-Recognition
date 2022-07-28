from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from models.last_epoch_writer_cb import LastEpochWriterCallback

def get_callbacks(checkpoints_dir, logdir, last_epoch_file_path, patience):
  return [
    ModelCheckpoint(filepath=checkpoints_dir),
    EarlyStopping(patience=patience),
    TensorBoard(logdir),
    LastEpochWriterCallback(last_epoch_file_path)
  ]
