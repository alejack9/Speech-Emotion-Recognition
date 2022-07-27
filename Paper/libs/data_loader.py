from os import listdir
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import tensorflow_io as tfio
import libs.data_operations as data_ops
import logging
import librosa
from consts import SEED

def get_dataset_information(samples_location, file_label_getter, file_speaker_getter):
  
  filenames = [f'{samples_location}/{p}' for p in listdir(samples_location)]
  labels = list(map(file_label_getter, filenames))


  df = pd.DataFrame({
    'filename': filenames,
    'speaker': map(file_speaker_getter, filenames),
    'label': labels})
  df['length'] = df['filename'].map(lambda x: librosa.get_duration(filename=x))
  
  logging.debug("Calculating max sample rate...")
  max_sample_rate = np.max(list(map(lambda file: tf.audio.decode_wav(contents=tf.io.read_file(file))[1].numpy(), filenames)))
  logging.debug("Calculating max sample rate... Done")

  logging.debug("Adding dummies (one-hot encoder)...")
  df = pd.get_dummies(df['label'], prefix="label").join(df)
  logging.debug("Adding dummies (one-hot encoder)... Done")

  one_hot_column_names = [col for col in df if col.startswith('label_')]

  one_hot_mapper = dict([(str(list(v[:-1])).replace(']', '.]').replace(',','.'), v[-1])
    for v in df[[*one_hot_column_names, 'label']].value_counts().index.values])

  return df, one_hot_mapper, max_sample_rate

def load_datasets(df, max_sample_rate, audio_sample_seconds=8, train_val_test_percentages=[62.5, 20.833, 16.666]):
  
  one_hot_column_names = [col for col in df if col.startswith('label_')]

  test_files, test_labels = np.array([]), np.array([])
  train_files, train_labels = np.array([]), np.array([])
  val_files, val_labels = np.array([]), np.array([])

  n_speakers = len(df['speaker'].unique())

  # Add samples splitting the duration

  df['start'] = df['length'].map(lambda x: np.arange(0, x, audio_sample_seconds))
  df['end'] = df['start'].map(lambda x: x + audio_sample_seconds)
  df = df.explode(['start', 'end'])
  # if length - start >= audio_sample_seconds / 2 => tieni
  df = df[df["length"] - df['start'] >= audio_sample_seconds / 2]


  per_speaker_test_samples = int(round(len(df) * train_val_test_percentages[2] / 100.0, 0) // n_speakers)
  per_speaker_val_samples = int(round(len(df) * train_val_test_percentages[1] / 100.0, 0) // n_speakers)

  trainval_test_splitter = StratifiedShuffleSplit(n_splits=10, test_size=per_speaker_test_samples, random_state=SEED)
  train_val_splitter = StratifiedShuffleSplit(n_splits=10, test_size=per_speaker_val_samples, random_state=SEED)

  for speaker in df['speaker'].unique():
    data = df[df['speaker'] == speaker]
    for train_val_index, test_index in trainval_test_splitter.split(data['filename'].to_numpy(), data['label'].to_numpy()):
      train_val_candidate_f = data['filename'].to_numpy()[train_val_index]
      train_val_candidate_l = data[one_hot_column_names].to_numpy()[train_val_index]
      test_candidate_f = data['filename'].to_numpy()[test_index]
      test_candidate_l = data[one_hot_column_names].to_numpy()[test_index]

    for train_index, val_index in train_val_splitter.split(train_val_candidate_f, train_val_candidate_l):
      train_candidate_f = train_val_candidate_f[train_index]
      train_candidate_l = train_val_candidate_l[train_index]
      val_candidate_f = train_val_candidate_f[val_index]
      val_candidate_l = train_val_candidate_l[val_index]

    test_files = np.append(test_files, test_candidate_f)
    test_labels = np.append(test_labels, test_candidate_l)

    train_files = np.append(train_files, train_candidate_f)
    train_labels = np.append(train_labels, train_candidate_l)

    val_files = np.append(val_files, val_candidate_f)
    val_labels = np.append(val_labels, val_candidate_l)

  test_labels = np.reshape(test_labels, (-1, len(one_hot_column_names)))
  train_labels = np.reshape(train_labels, (-1, len(one_hot_column_names)))
  val_labels = np.reshape(val_labels, (-1, len(one_hot_column_names)))

  logging.info(f'Training set size: {len(train_files)}')
  logging.info(f'Validation set size: {len(val_files)}')
  logging.info(f'Test set size: {len(test_files)}')

  logging.debug('Train sample:')
  logging.debug(f'({train_files[0]}, {train_labels[0]})')
  logging.debug('Val sample:')
  logging.debug(f'({val_files[0]}, {val_labels[0]})')
  logging.debug('Test sample:')
  logging.debug(f'({test_files[0]}, {test_labels[0]})')

  operations = [
    data_ops.ReadFile(),
    data_ops.DecodeWav(with_sample_rate=True),
    data_ops.Resample(max_sample_rate),
    data_ops.Squeeze(),
    data_ops.Crop(0, max_sample_rate * audio_sample_seconds),
    data_ops.ZeroPad(max_sample_rate * audio_sample_seconds),
    data_ops.CastToFloat(),
    data_ops.Reshape((max_sample_rate * audio_sample_seconds, 1)),
  ]

  train_ds = tfio.audio.AudioIODataset.from_tensor_slices((train_files, train_labels))
  val_ds = tfio.audio.AudioIODataset.from_tensor_slices((val_files, val_labels))
  test_ds = tfio.audio.AudioIODataset.from_tensor_slices((test_files, test_labels))

  for o in operations:
    train_ds = train_ds.map(o, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(o, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(o, num_parallel_calls=tf.data.AUTOTUNE)

  return train_ds, val_ds, test_ds, {
    'labels_distribution': {
      'complete': np.unique(df['label'], return_counts=True),
      'train': np.unique(list(map(str, train_labels)), return_counts=True),
      'val': np.unique(list(map(str, val_labels)), return_counts=True)
    }
  }