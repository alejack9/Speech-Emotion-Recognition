from tensorflow import keras

import libs.data_loader as data_loader
import libs.data_operations as data_ops

import pandas as pd

# CROP_FADE_15  
data_preprocessing = lambda total_audio_frames: [
  data_ops.Crop(),
  data_ops.Fade(int(total_audio_frames * 0.15), int(total_audio_frames * 0.15)),
]

df = pd.read_pickle("./data/all.pkl")
df = df.astype({'label_d': 'int32', 'label_f': 'int32'})
_, _, test_ds, _ = data_loader.load_datasets(df, 44100, 3, data_preprocessing, [80, 10, 10])

test_ds = test_ds.batch(16)

SAVEE_model = keras.models.load_model("./saved_models/mPaperModel_s3_b16_d0_p80_o_crop_fade15_sz62.5,20.833,16.666/")
SAVEE_model.summary()

SAVEE_model.evaluate(test_ds)