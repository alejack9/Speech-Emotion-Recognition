from models.files_names_cb import FirstFilenameSetCallback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from models.last_epoch_writer_cb import LastEpochWriterCallback

def get_callbacks(checkpoints_dir, logdir, last_epoch_file_path, patience):
  return [
    ModelCheckpoint(filepath=checkpoints_dir),
    EarlyStopping(patience=patience),
    TensorBoard(logdir),
    # LastEpochWriterCallback(last_epoch_file_path),
    # FirstFilenameSetCallback(train_filenames, val_filenames, test_filenames)
  ]
