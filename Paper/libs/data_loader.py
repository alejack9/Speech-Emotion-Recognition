import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import tensorflow_io as tfio
import Paper.libs.data_operations as data_ops
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data(samples_location, file_label_getter, file_speaker_getter, audio_sample_seconds=8, train_val_test_sizes=[62.5, 20.833, 16.666]):
  seed = 42
  
  filenames = [f'{samples_location}/{p}' for p in os.listdir(samples_location)]
  labels = list(map(file_label_getter, filenames))


  df = pd.DataFrame({
    'filenames': filenames,
    'speaker': map(file_speaker_getter, filenames),
    'label': labels})
  
  max_sample_rate = np.max(list(map(lambda file: tf.audio.decode_wav(contents=tf.io.read_file(file))[1].numpy(), filenames)))
  
  df = pd.get_dummies(df['label'], prefix="label").join(df)

  one_hot_column_names = [col for col in df if col.startswith('label_')]
  test_files, test_labels = np.array([]), np.array([])
  train_files, train_labels = np.array([]), np.array([])
  val_files, val_labels = np.array([]), np.array([])

  n_speakers = len(df['speaker'].unique())

  trainval_test_shuff = StratifiedShuffleSplit(n_splits=10, test_size=int(train_val_test_sizes[2] / n_speakers), random_state=seed)
  train_val_shuff = StratifiedShuffleSplit(n_splits=10, test_size=int(train_val_test_sizes[1] / n_speakers), random_state=seed)

  for speaker in df['speaker'].unique():
    data = df[df['speaker'] == speaker]
    for train_val_index, test_index in trainval_test_shuff.split(data['filenames'].to_numpy(), data['label'].to_numpy()):
      train_val_candidate_f = data['filenames'].to_numpy()[train_val_index]
      train_val_candidate_l = data[one_hot_column_names].to_numpy()[train_val_index]
      test_candidate_f = data['filenames'].to_numpy()[test_index]
      test_candidate_l = data[one_hot_column_names].to_numpy()[test_index]

    for train_index, val_index in train_val_shuff.split(train_val_candidate_f, train_val_candidate_l):
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

  print('Training set size', len(train_files))
  print('Validation set size', len(val_files))
  print('Test set size', len(test_files))

  print(train_files[0], train_labels[0])
  print(val_files[0], val_labels[0])
  print(test_files[0], test_labels[0])

  operations = [
    data_ops.ReadFile(),
    data_ops.DecodeWav(with_sample_rate=True),
    data_ops.Resample(max_sample_rate),
    data_ops.Squeeze(),
    data_ops.Crop(max_sample_rate * audio_sample_seconds),
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

  return train_ds, val_ds, test_ds, {'origin_df': df,
    'labels_distribution': {
      'complete': np.unique(labels, return_counts=True),
      'train': np.unique(list(map(str, train_labels)), return_counts=True),
      'val': np.unique(list(map(str, val_labels)), return_counts=True)
    }
  }