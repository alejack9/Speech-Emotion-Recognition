from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from libs.data_loader import load_data
import libs.model_runner as model_runner
import tensorflow as tf
import numpy as np
import datetime
from paper_model import PaperModel
from consts import DATA_DIR, SEED, CHECKPOINT_DIR, LOGS_DIR, LAST_EPOCH_FILE
import os
import re 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set the seed value for experiment reproducibility.
tf.random.set_seed(SEED)
np.random.seed(SEED)

def get_label(file_path):
    parts = re.sub('.+\_|[0-9]+.wav', '', file_path)
    return parts


def get_speaker_name(file_path):
    parts = re.sub('.*[/]+|\_|[a-z]+[0-9]+.wav', '', file_path)
    return parts

train_ds, val_ds, test_ds, additional = load_data(DATA_DIR, get_label, get_speaker_name, audio_sample_seconds=8, train_val_test_sizes=[300, 100, 80])

class LastEpochWriterCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(LAST_EPOCH_FILE, "w") as f:
            f.write(f"{epoch}")

model = PaperModel().get_model()

train_ds = train_ds.batch(8)
val_ds = val_ds.batch(8)

EPOCHS = 400

init_epoch = 0

if os.path.exists(LAST_EPOCH_FILE):
    model.load_weights(CHECKPOINT_DIR)
    with open(LAST_EPOCH_FILE, "r") as f:
        init_epoch = int(f.readline())

checkpoint_callback = ModelCheckpoint(filepath=CHECKPOINT_DIR)
logdir = os.path.join(LOGS_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = TensorBoard(logdir)
last_epoch_writer_cb = LastEpochWriterCallback()

model_runner.run(model, train_ds, val_ds, cbs=[checkpoint_callback, tensorboard_callback, last_epoch_writer_cb], epochs=EPOCHS, init_epoch=init_epoch)