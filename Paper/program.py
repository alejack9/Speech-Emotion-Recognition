from itertools import product
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import libs.data_loader as data_loader
import libs.model_runner as model_runner
import libs.data_visualization as visual
from libs.last_epoch_writer_cb import LastEpochWriterCallback
import hyperparams
from os.path import join
import datetime
import logging
import re
import os
import librosa

def create_folder(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

def get_label(file_path):
  parts = re.sub('.+\_|[0-9]+.wav', '', file_path)
  return parts


def get_speaker_name(file_path):
  parts = re.sub('.*[/]+|\_|[a-z]+[0-9]+.wav', '', file_path)
  return parts

def data_analysis(plots_dir, df):
  visual.plot_speakers_pie(df, output_file=join(plots_dir, "speakers_pie.png"))
  visual.kde_plot(df['filenames'].map(lambda x: librosa.get_duration(filename=x)).rename("lengths") , output_file=join(plots_dir, "lengths_kde.png"))

def run(data_dir, working_dir, epochs, batch_sizes):
  plots_dir = create_folder(join(working_dir, "plots"))

  logging.info("Loading data...")
  df, one_hot_mapper, max_sample_rate = data_loader.get_dataset_information(data_dir, get_label, get_speaker_name)
  logging.info("Loading data... done")

  data_analysis(plots_dir, df)

  for model_factory, train_val_tests_size, seconds in product(hyperparams.model_factories, hyperparams.train_val_test_sizes, hyperparams.seconds):
    selected_batch_size_index = -1
    computed = False
    while not computed:
      if selected_batch_size_index < len(batch_sizes):
        selected_batch_size_index = selected_batch_size_index + 1
        batch_size = batch_sizes[selected_batch_size_index]
      else:
        logging.critical("No batch size is suitable for this hardware. Trying next hyper-parameters configuration.")
        break

      logging.info("-------------------------")
      logging.info(f'Model: {model_factory.get_model_name()} , seconds: {seconds} , batch_size: {batch_size}')

      train_ds, val_ds, test_ds, additional = data_loader.load_datasets(df, max_sample_rate, audio_sample_seconds=seconds,
        train_val_test_sizes=train_val_tests_size)

      model = model_factory.get_model(args={"input_shape": (max_sample_rate * seconds, 1), 'print_summary': False})
      model_name = f"m{model_factory.get_model_name()}_s{seconds}_b{batch_size}_sizes{str(train_val_tests_size).replace(' ', '')[1:-1]}"
      
      current_plots_dir = create_folder(join(plots_dir, model_name))
      checkpoints_dir = create_folder(join(working_dir, "checkpoints", model_name))
      logs_dir = create_folder(join(working_dir, "logs", model_name))
      last_epoch_file_path = create_folder(join(working_dir, "last_epochs")) + f"{model_name}.txt"

      visual.plot_labels_distribution(additional['labels_distribution'], one_hot_mapper, output_file=join(current_plots_dir, "labels_distribution.png"))
      visual.plot_audio_waves(train_ds, one_hot_mapper, output_file=join(current_plots_dir, "audio_waves.png"))

      train_ds = train_ds.batch(batch_size)
      val_ds = val_ds.batch(batch_size)

      init_epoch = 0

      if os.path.exists(last_epoch_file_path):
        model.load_weights(checkpoints_dir)
        with open(last_epoch_file_path, "r") as f:
          init_epoch = int(f.readline()) + 1

      logdir = join(logs_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

      checkpoint_cb = ModelCheckpoint(filepath=checkpoints_dir)
      early_stopping_cb = EarlyStopping(patience=25)
      tensorboard_cb = TensorBoard(logdir)
      last_epoch_writer_cb = LastEpochWriterCallback(last_epoch_file_path)

      try:
        model_runner.run(model, train_ds, val_ds, cbs=[checkpoint_cb, tensorboard_cb, last_epoch_writer_cb, early_stopping_cb], 
          epochs=epochs, init_epoch=init_epoch)
        computed = True
      except ResourceExhaustedError as e:
        logging.error("Not enough GPU memory.")
        # logging.error(e)