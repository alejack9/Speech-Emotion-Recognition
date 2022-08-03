import libs.data_operations as data_ops
from models.paper_model import PaperModelFactory
import libs.data_operations as data_ops
from functools import reduce

combinations = {
  'model_factories': [
    PaperModelFactory()
  ],
  'seconds' : [3, 4, 5, 8],
  'patiences' : [25, 80],
  'dropouts': [0, 0.2, 0.5],
  'train_val_test_percentages' : [(62.5, 20.833, 16.666)],
  'data_operations_factories' : [
    ('crop', lambda _: [
        data_ops.Crop(),
    ]),
    ('fade05_crop', lambda total_audio_frames: [
      data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
      data_ops.Crop(),
    ]),
    ('fade10_crop', lambda total_audio_frames: [
      data_ops.Fade(total_audio_frames * 0.10, total_audio_frames * 0.10),
      data_ops.Crop(),
    ]),
    ('fade15_crop', lambda total_audio_frames: [
      data_ops.Fade(total_audio_frames * 0.15, total_audio_frames * 0.15),
      data_ops.Crop(),
    ]),
    ('crop_fade05', lambda total_audio_frames: [
      data_ops.Crop(),
      data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
    ]),
    ('crop_fade10', lambda total_audio_frames: [
      data_ops.Crop(),
      data_ops.Fade(total_audio_frames * 0.10, total_audio_frames * 0.10),
    ]),
    ('crop_fade15', lambda total_audio_frames: [
      data_ops.Crop(),
      data_ops.Fade(total_audio_frames * 0.15, total_audio_frames * 0.15),
    ]),
    ('crop_norm', lambda _: [
        data_ops.Crop(),
        data_ops.Normalize()
    ]),
    ('fade05_crop_norm', lambda total_audio_frames: [
      data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
      data_ops.Crop(),
      data_ops.Normalize()
    ]),
    ('fade10_crop_norm', lambda total_audio_frames: [
      data_ops.Fade(total_audio_frames * 0.10, total_audio_frames * 0.10),
      data_ops.Crop(),
      data_ops.Normalize()
    ]),
    ('fade15_crop_norm', lambda total_audio_frames: [
      data_ops.Fade(total_audio_frames * 0.15, total_audio_frames * 0.15),
      data_ops.Crop(),
      data_ops.Normalize()
    ]),
    ('crop_fade05_norm', lambda total_audio_frames: [
      data_ops.Crop(),
      data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
      data_ops.Normalize()
    ]),
    ('crop_fade10_norm', lambda total_audio_frames: [
      data_ops.Crop(),
      data_ops.Fade(total_audio_frames * 0.10, total_audio_frames * 0.10),
      data_ops.Normalize()
    ]),
    ('crop_fade15_norm', lambda total_audio_frames: [
      data_ops.Crop(),
      data_ops.Fade(total_audio_frames * 0.15, total_audio_frames * 0.15),
      data_ops.Normalize()
    ]),
  ]
}

total = reduce(lambda a, b: a * b, list(map(len, combinations.values())), 1)
