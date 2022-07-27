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

  one_hot_mapper = dict([(str(list(v[:-1])).replace(',',''), v[-1])
    for v in df[[*one_hot_column_names, 'label']].value_counts().index.values])

  return df, one_hot_mapper, max_sample_rate

def load_datasets(df, max_sample_rate, audio_sample_seconds=8, train_val_test_percentages=[62.5, 20.833, 16.666]):
  
  one_hot_column_names = [col for col in df if col.startswith('label_')]

  n_speakers = len(df['speaker'].unique())

  # Add samples splitting the duration

  df['start'] = df['length'].map(lambda x: np.arange(0, x, audio_sample_seconds))
  df['end'] = df['start'].map(lambda x: x + audio_sample_seconds)
  df = df.explode(['start', 'end'])
  # if length - start >= audio_sample_seconds / 2 => tieni
  df = df[df["length"] - df['start'] >= audio_sample_seconds / 2]
  df['start'] = df['start'].map(lambda x: x * max_sample_rate)
  df['end'] = df['end'].map(lambda x: x * max_sample_rate)


  per_speaker_test_samples = int(round(len(df) * train_val_test_percentages[2] / 100.0, 0) // n_speakers)
  per_speaker_val_samples = int(round(len(df) * train_val_test_percentages[1] / 100.0, 0) // n_speakers)

  trainval_test_splitter = StratifiedShuffleSplit(n_splits=10, test_size=per_speaker_test_samples, random_state=SEED)
  train_val_splitter = StratifiedShuffleSplit(n_splits=10, test_size=per_speaker_val_samples, random_state=SEED)

  train_ds, val_ds, test_ds = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

  for speaker in df['speaker'].unique():
    data = df[df['speaker'] == speaker]

    for train_val_indexes, test_indexes in trainval_test_splitter.split(np.zeros(len(data)), data['label']):
      speaker_train_val_i = train_val_indexes
      speaker_test_ds = data.iloc[test_indexes, :]

    for train_indexes, val_indexes in train_val_splitter.split(speaker_train_val_i, data['label'].iloc[speaker_train_val_i]):
      speaker_train_ds = data.iloc[train_indexes, :]
      speaker_val_ds = data.iloc[val_indexes, :]
      
    train_ds = pd.concat([train_ds, speaker_train_ds])
    val_ds = pd.concat([val_ds, speaker_val_ds])
    test_ds = pd.concat([test_ds, speaker_test_ds])
    
  logging.info(f'Training set size: {len(train_ds)}')
  logging.info(f'Validation set size: {len(val_ds)}')
  logging.info(f'Test set size: {len(test_ds)}')

  logging.debug('Train sample:')
  logging.debug(f"({train_ds['filename'].iloc[0]}, {train_ds[one_hot_column_names].iloc[0]})")
  logging.debug('Val sample:')
  logging.debug(f"({val_ds['filename'].iloc[0]}, {val_ds[one_hot_column_names].iloc[0]})")
  logging.debug('Test sample:')
  logging.debug(f"({test_ds['filename'].iloc[0]}, {test_ds[one_hot_column_names].iloc[0]})")

  train_tf_ds = tfio.audio.AudioIODataset.from_tensor_slices((
    train_ds['filename'].to_numpy(),
    train_ds[one_hot_column_names].to_numpy(),
    np.asarray(train_ds['start'].to_numpy()).astype('int32'),
    np.asarray(train_ds['end'].to_numpy()).astype('int32')
    ))
  val_tf_ds = tfio.audio.AudioIODataset.from_tensor_slices((
    val_ds['filename'].to_numpy(),
    val_ds[one_hot_column_names].to_numpy(),
    np.asarray(val_ds['start'].to_numpy()).astype('int32'),
    np.asarray(val_ds['end'].to_numpy()).astype('int32')
    ))
  test_tf_ds = tfio.audio.AudioIODataset.from_tensor_slices((
    test_ds['filename'].to_numpy(),
    test_ds[one_hot_column_names].to_numpy(),
    np.asarray(test_ds['start'].to_numpy()).astype('int32'),
    np.asarray(test_ds['end'].to_numpy()).astype('int32')
    ))

  total_audio_frames = max_sample_rate * audio_sample_seconds

  operations = [
    data_ops.ReadFile(),
    data_ops.DecodeWav(with_sample_rate=True),
    data_ops.Resample(max_sample_rate),
    data_ops.Squeeze(),
    data_ops.Trim(),
    data_ops.Crop(),
    data_ops.Fade(total_audio_frames * 0.005, total_audio_frames * 0.005),
    data_ops.ZeroPad(total_audio_frames),
    data_ops.CastToFloat(),
    data_ops.Reshape((total_audio_frames, 1)),
    data_ops.RemoveCropInformation()
  ]

  for o in operations:
    train_tf_ds = train_tf_ds.map(o, num_parallel_calls=tf.data.AUTOTUNE)
    val_tf_ds = val_tf_ds.map(o, num_parallel_calls=tf.data.AUTOTUNE)
    test_tf_ds = test_tf_ds.map(o, num_parallel_calls=tf.data.AUTOTUNE)

  return train_tf_ds, val_tf_ds, test_tf_ds, {
    'labels_distribution': {
      'complete': np.unique(df['label'], return_counts=True),
      'train': np.unique(list(map(str, train_ds['label'])), return_counts=True),
      'val': np.unique(list(map(str, val_ds['label'])), return_counts=True)
    }
  }
