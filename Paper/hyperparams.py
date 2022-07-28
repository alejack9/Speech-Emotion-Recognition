from models.paper_model import PaperModelFactory
import libs.data_operations as data_ops

model_factories = [
  PaperModelFactory()
]

seconds = [3, 4, 5, 8]

train_val_test_percentages = [(62.5, 20.833, 16.666)]

data_operations_factories = [
  ('trim', lambda _: [
    data_ops.Trim(),
  ]),
  # ('fade05', lambda total_audio_frames: [
  #   data_ops.CastToFloat(),
  #   data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
  # ]),
  # ('trim_fade05', lambda total_audio_frames: [
  #   data_ops.Trim(),
  #   data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
  # ]),
  # ('trim_crop_fade05', lambda total_audio_frames: [
  #   data_ops.Trim(),
  #   data_ops.Crop(),
  #   data_ops.Fade(round(total_audio_frames * 0.05), round(total_audio_frames * 0.05)),
  # ])
]
