from itertools import product
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import libs.data_loader as data_loader
import libs.model_runner as model_runner
import libs.data_visualization as visual
from libs.utils import create_folder
from models.callbacks import get_callbacks
from hyperparams import total as total_pars, combinations as hyperparams
from os.path import join
from datetime import datetime
import logging
import re
import os

def get_label(file_path):
  parts = re.sub('.+\_|[0-9]+.wav', '', file_path)
  return parts


def get_speaker_name(file_path):
  parts = re.sub('.*[/]+|\_|[a-z]+[0-9]+.wav', '', file_path)
  return parts

def data_analysis(plots_dir, df):
  visual.plot_speakers_pie(df, output_file=join(plots_dir, "speakers_pie.png"))
  visual.kde_plot(df['length'], output_file=join(plots_dir, "lengths_kde.png"))

def run(data_dir, working_dir, epochs, batch_sizes):

  plots_dir = create_folder(join(working_dir, "plots"))
  completed_models_dir = create_folder(join(working_dir, "done"))

  logging.info("Loading data...")
  df, one_hot_mapper, max_sample_rate = data_loader.get_dataset_information(data_dir, get_label, get_speaker_name)
  logging.info("Loading data... done")

  data_analysis(plots_dir, df)

  for i, (model_factory, train_val_tests_percentage, audio_seconds, (data_ops_name, data_ops_factory), patience) in enumerate(product(hyperparams['model_factories'], hyperparams['train_val_test_percentages'], hyperparams['seconds'], hyperparams['data_operations_factories'], hyperparams['patiences'])):
    computed = False
    
    for j, batch_size in enumerate(batch_sizes):
      logging.info(f"------------- Trying batch size {j + 1} / {len(batch_sizes)} ({round((j + 1) / float(len(batch_sizes)) * 100, 2)}%) ------------")
      logging.info(f'Model: {model_factory.get_model_name()} , seconds: {audio_seconds} , batch_size: {batch_size} , early_stopping_patience: {patience} data_ops: {data_ops_name}')

      train_ds, val_ds, test_ds, additional = data_loader.load_datasets(df,
        max_sample_rate, audio_seconds,
        data_ops_factory, train_val_tests_percentage)

      model = model_factory.get_model(args={"input_shape": (max_sample_rate * audio_seconds, 1), 'print_summary': False})
      model_name = f"m{model_factory.get_model_name()}_s{audio_seconds}_b{batch_size}_p{patience}_o_{data_ops_name}_sz{str(train_val_tests_percentage).replace(' ', '')[1:-1]}"

      done_model_path = join(completed_models_dir, model_name)

      if os.path.exists(done_model_path):
        logging.info("Model already computed.")
        computed = True
        break
      
      current_plots_dir = create_folder(join(plots_dir, model_name))
      checkpoints_dir = create_folder(join(working_dir, "checkpoints", model_name))
      logs_dir = create_folder(join(working_dir, "logs", model_name, datetime.now().strftime("%Y%m%d-%H%M%S")))
      last_epoch_file_path = join(create_folder(join(working_dir, "last_epochs")), f"{model_name}.txt")

      visual.plot_labels_distribution(additional['labels_distribution'], one_hot_mapper, output_file=join(current_plots_dir, "labels_distribution.png"))
      visual.plot_audio_waves(train_ds, one_hot_mapper, output_file=join(current_plots_dir, "audio_waves.png"))

      train_ds = train_ds.batch(batch_size)
      val_ds = val_ds.batch(batch_size)

      init_epoch = 0

      if os.path.exists(last_epoch_file_path):
        model.load_weights(checkpoints_dir)
        with open(last_epoch_file_path, "r") as f:
          init_epoch = int(f.readline())

      logdir = join(logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))

      cbs = get_callbacks(checkpoints_dir, logdir, last_epoch_file_path, patience)

      try:
        model_runner.run(model, train_ds, val_ds, cbs=cbs, epochs=epochs, init_epoch=init_epoch)
        computed = True
      except ResourceExhaustedError:
        logging.error("Not enough GPU memory.")

      if computed == True:
        open(done_model_path, "w+").close()

    if not computed:
      logging.critical("No batch size is suitable for this hardware. Trying next hyper-parameters configuration.")

    logging.info(f"-------------------- {str(i + 1).rjust(len(str(total_pars)), ' ')} / {total_pars} ({round((i + 1) / float(total_pars) * 100, 2)}%) --------------------")
